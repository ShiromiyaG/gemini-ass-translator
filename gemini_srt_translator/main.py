# gemini_srt_translator.py

import datetime
import json
import os
import signal
import sys
import time
import typing
import unicodedata as ud
import copy
from collections import Counter
import subprocess

import json_repair
import pysubs2
from google import genai
from google.genai import types
from google.genai.types import Content
from srt import Subtitle

from gemini_srt_translator.logger import (
    error,
    error_with_progress,
    get_last_chunk_size,
    highlight,
    highlight_with_progress,
    info,
    info_with_progress,
    input_prompt,
    input_prompt_with_progress,
    progress_bar,
    save_logs_to_file,
    save_thoughts_to_file,
    set_color_mode,
    success_with_progress,
    update_loading_animation,
    warning,
    warning_with_progress,
)

from .ffmpeg_utils import (
    check_ffmpeg_installation,
    extract_srt_from_video,
    prepare_audio,
)
from .helpers import get_instruction, get_response_schema, get_safety_settings


def extract_json_from_text(text: str) -> str:
    """
    Extracts a JSON object or array from a string.
    It handles cases where the JSON is embedded within other text.
    """
    # Remove markdown fences and other non-JSON text
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

    # Find the start of the JSON array or object
    start_pos = text.find('[')
    if start_pos == -1:
        start_pos = text.find('{')

    # Find the end of the JSON array or object
    end_pos = text.rfind(']')
    if end_pos == -1:
        end_pos = text.rfind('}')

    if start_pos != -1 and end_pos != -1:
        return text[start_pos:end_pos+1]

    return text # Return original text if no JSON structure is found


class SubtitleObject(typing.TypedDict):
    """
    TypedDict for subtitle objects used in translation
    """

    index: str
    content: str
    time_start: typing.Optional[str] = None
    time_end: typing.Optional[str] = None


class GeminiSRTTranslator:
    """
    A translator class that uses Gemini API to translate subtitles.
    """

    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_api_key2: str = None,
        target_language: str = None,
        input_file: str = None,
        output_file: str = None,
        video_file: str = None,
        audio_file: str = None,
        extract_audio: bool = False,
        start_line: int = None,
        description: str = None,
        model_name: str = "gemini-2.5-pro",
        batch_size: int = 300,
        streaming: bool = True,
        thinking: bool = True,
        thinking_budget: int = 2048,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        free_quota: bool = True,
        use_colors: bool = True,
        progress_log: bool = False,
        thoughts_log: bool = False,
        resume: bool = None,
        debug: bool = False,
        preserve_format: bool = True,  # <--- novo parâmetro
    ):
        """
        Initialize the translator with necessary parameters.

        Args:
            gemini_api_key (str): Primary Gemini API key
            gemini_api_key2 (str): Secondary Gemini API key for additional quota
            target_language (str): Target language for translation
            input_file (str): Path to input subtitle file
            output_file (str): Path to output translated subtitle file
            video_file (str): Path to video file for srt/audio extraction
            audio_file (str): Path to audio file for translation
            extract_audio (bool): Whether to extract audio from video for translation
            start_line (int): Line number to start translation from
            description (str): Additional instructions for translation
            model_name (str): Gemini model to use
            batch_size (int): Number of subtitles to process in each batch
            streaming (bool): Whether to use streamed responses
            thinking (bool): Whether to use thinking mode
            thinking_budget (int): Budget for thinking mode
            free_quota (bool): Whether to use free quota (affects rate limiting)
            use_colors (bool): Whether to use colored output
            progress_log (bool): Whether to log progress to a file
            thoughts_log (bool): Whether to log thoughts to a file
        """
        self.debug = debug

        base_file = input_file or video_file
        base_name = os.path.splitext(os.path.basename(base_file))[0] if base_file else "translated"
        dir_path = os.path.dirname(base_file) if base_file else ""

        self.log_file_path = (
            os.path.join(dir_path, f"{base_name}.progress.log") if dir_path else f"{base_name}.progress.log"
        )
        self.thoughts_file_path = (
            os.path.join(dir_path, f"{base_name}.thoughts.log") if dir_path else f"{base_name}.thoughts.log"
        )
        self.debug_log_file = (
            os.path.join(dir_path, "gemini_translator_debug.log") if dir_path else "gemini_translator_debug.log"
        )

        if output_file:
            self.output_file = output_file
        else:
            suffix = "_translated.ass"
            self.output_file = os.path.join(dir_path, f"{base_name}{suffix}") if dir_path else f"{base_name}.ass"

        self.progress_file = os.path.join(dir_path, f"{base_name}.progress") if dir_path else f"{base_name}.progress"

        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.target_language = target_language
        self.input_file = input_file
        self.video_file = video_file
        self.audio_file = audio_file
        self.extract_audio = extract_audio
        self.start_line = start_line
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.streaming = streaming
        self.thinking = thinking
        self.thinking_budget = thinking_budget if thinking else 0
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.free_quota = free_quota
        self.progress_log = progress_log
        self.thoughts_log = thoughts_log
        self.resume = resume
        self.preserve_format = preserve_format
        self.original_ext = None  # será definido depois que input_file for resolvido

        self.current_api_number = 1
        self.backup_api_number = 2
        self.batch_number = 1
        self.audio_part = None
        self.token_limit = 0
        self.token_count = 0
        self.translated_batch = []
        self.srt_extracted = False
        self.audio_extracted = False
        self.ffmpeg_installed = check_ffmpeg_installation()

        # Set color mode based on user preference
        set_color_mode(use_colors)

        if self.debug:
            self._log_debug("--- Debug Mode Enabled ---")
            self._log_debug(f"Time: {datetime.datetime.now()}")
            self._log_debug(f"Initial parameters: {self.__dict__}")

    def _log_debug(self, message: str):
        if not self.debug:
            return
        with open(self.debug_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")

    def _get_config(self):
        """Get the configuration for the translation model."""
        thinking_compatible = False
        thinking_budget_compatible = False
        if "2.5" in self.model_name:
            thinking_compatible = True
        if "flash" in self.model_name:
            thinking_budget_compatible = True

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=get_response_schema(),
            safety_settings=get_safety_settings(),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            system_instruction=get_instruction(
                language=self.target_language,
                thinking=self.thinking,
                thinking_compatible=thinking_compatible,
                audio_file=self.audio_file,
                description=self.description,
            ),
            thinking_config=(
                types.ThinkingConfig(
                    include_thoughts=self.thinking,
                    thinking_budget=self.thinking_budget if thinking_budget_compatible else None,
                )
                if thinking_compatible
                else None
            ),
        )

    def _check_saved_progress(self):
        """Check if there's a saved progress file and load it if exists"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return

        if self.start_line != None:
            return

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                saved_line = data.get("line", 1)
                input_file = data.get("input_file")

                # Verify the progress file matches our current input file
                if input_file != self.input_file:
                    warning(f"Found progress file for different subtitle: {input_file}")
                    warning("Ignoring saved progress.")
                    return

                if saved_line > 1:
                    if self.resume is None:
                        resume = input_prompt(f"Found saved progress. Resume? (y/n): ", mode="resume").lower().strip()
                    elif self.resume is True:
                        resume = "y"
                    elif self.resume is False:
                        resume = "n"
                    if resume == "y" or resume == "yes":
                        info(f"Resuming from line {saved_line}")
                        self.start_line = saved_line
                    else:
                        info("Starting from the beginning")
                        # Remove the progress file
                        try:
                            os.remove(self.output_file)
                        except Exception as e:
                            pass
        except Exception as e:
            warning(f"Error reading progress file: {e}")

    def _save_progress(self, line):
        """Save current progress to temporary file"""
        if not self.progress_file:
            return

        try:
            with open(self.progress_file, "w") as f:
                json.dump({"line": line, "input_file": self.input_file}, f)
        except Exception as e:
            warning_with_progress(f"Failed to save progress: {e}")

    def getmodels(self):
        """Get available Gemini models that support content generation."""
        if not self.current_api_key:
            error("Please provide a valid Gemini API key.")
            exit(1)

        client = self._get_client()
        models = client.models.list()
        list_models = []
        for model in models:
            supported_actions = model.supported_actions
            if "generateContent" in supported_actions:
                list_models.append(model.name.replace("models/", ""))
        return list_models

    def translate(self):
        """
        Main translation method. Reads the input subtitle file, translates it in batches,
        and writes the translated subtitles to the output file.
        """

        if not self.ffmpeg_installed and self.video_file:
            error("FFmpeg is not installed. Please install FFmpeg to use video features.", ignore_quiet=True)
            exit(1)

        if self.video_file and self.extract_audio:
            if os.path.exists(self.video_file):
                self.audio_file = prepare_audio(self.video_file)
                self.audio_extracted = True
            else:
                error(f"Video file {self.video_file} does not exist.", ignore_quiet=True)
                exit(1)

        if self.audio_file:
            if os.path.exists(self.audio_file):
                with open(self.audio_file, "rb") as f:
                    audio_bytes = f.read()
                    self.audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
            else:
                error(f"Audio file {self.audio_file} does not exist.", ignore_quiet=True)
                exit(1)

        if self.video_file and not self.input_file:
            # (extração de legendas)
            if not os.path.exists(self.video_file):
                error(f"Video file {self.video_file} does not exist.", ignore_quiet=True)
                exit(1)
            self.input_file = extract_srt_from_video(self.video_file, ".ass")
            if not self.input_file:
                error("Failed to extract subtitles from video file. Only .ass format is supported for extraction.", ignore_quiet=True)
                exit(1)
            self.srt_extracted = True

        # Definir extensão original após garantir self.input_file
        if self.input_file and not self.original_ext:
            self.original_ext = os.path.splitext(self.input_file)[1].lower()

        # Gerar output_file padrão mantendo a extensão original se solicitado
        if not self.output_file:
            base_name, _ext = os.path.splitext(self.input_file)
            self.output_file = f"{base_name}_translated.ass"

        # Se o usuário passou um output_file mas queremos preservar .ass
        if (
            self.preserve_format
            and self.original_ext == ".ass"
            and not self.output_file.lower().endswith(".ass")
        ):
            root, _ = os.path.splitext(self.output_file)
            self.output_file = root + ".ass"

        if not self.current_api_key:
            error("Please provide a valid Gemini API key.", ignore_quiet=True)
            exit(1)

        if not self.target_language:
            error("Please provide a target language.", ignore_quiet=True)
            exit(1)

        if self.input_file and not os.path.exists(self.input_file):
            error(f"Input file {self.input_file} does not exist.", ignore_quiet=True)
            exit(1)

        elif not self.input_file:
            error("Please provide a subtitle or video file.", ignore_quiet=True)
            exit(1)

        if self.thinking_budget < 0 or self.thinking_budget > 24576:
            error("Thinking budget must be between 0 and 24576. 0 disables thinking.", ignore_quiet=True)
            exit(1)

        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            error("Temperature must be between 0.0 and 2.0.", ignore_quiet=True)
            exit(1)

        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            error("Top P must be between 0.0 and 1.0.", ignore_quiet=True)
            exit(1)

        if self.top_k is not None and self.top_k < 0:
            error("Top K must be a non-negative integer.", ignore_quiet=True)
            exit(1)

        self._check_saved_progress()

        models = self.getmodels()

        if self.model_name not in models:
            error(f"Model {self.model_name} is not available. Please choose a different model.", ignore_quiet=True)
            exit(1)

        self._get_token_limit()

        translated_file = None  # <--- Inicializa como None aqui

        with open(self.input_file, "r", encoding="utf-8-sig") as original_file:
            original_subtitle = pysubs2.load(self.input_file, encoding="utf-8-sig")
            total = len(original_subtitle)
            if not original_subtitle:
                error("Subtitle file is empty.")
                return
            try:
                translated_file_exists = open(self.output_file, "r", encoding="utf-8")
                translated_subtitle = pysubs2.load(translated_file_exists.read())
                info(f"Translated file {self.output_file} already exists. Loading existing translation...\n")
                if self.start_line == None:
                    while True:
                        try:
                            self.start_line = int(
                                input_prompt(
                                    f"Enter the line number to start from (1 to {len(original_subtitle)}): ",
                                    mode="line",
                                    max_length=len(original_subtitle),
                                ).strip()
                            )
                            if self.start_line < 1 or self.start_line > len(original_subtitle):
                                warning(
                                    f"Line number must be between 1 and {len(original_subtitle)}. Please try again."
                                )
                                continue
                            break
                        except ValueError:
                            warning("Invalid input. Please enter a valid number.")

            except FileNotFoundError:
                translated_subtitle = copy.deepcopy(original_subtitle)
                self.start_line = 1

            if len(original_subtitle) != len(translated_subtitle):
                error(
                    f"Number of lines of existing translated file does not match the number of lines in the original file.",
                    ignore_quiet=True,
                )
                exit(1)

            translated_file = open(self.output_file, "w", encoding="utf-8")

            if self.start_line > len(original_subtitle) or self.start_line < 1:
                error(
                    f"Start line must be between 1 and {len(original_subtitle)}. Please try again.", ignore_quiet=True
                )
                exit(1)

            if len(original_subtitle) < self.batch_size:
                self.batch_size = len(original_subtitle)

            delay = False
            delay_time = 30

            if "pro" in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    info("Pro model and free user quota detected.\n")
                else:
                    delay_time = 15
                    info("Pro model and free user quota detected, using secondary API key if needed.\n")

            i = self.start_line
            total = len(original_subtitle)
            batch = []
            previous_message = []
            if self.start_line > 1:
                start_idx = max(0, self.start_line - 2 - self.batch_size)
                start_time = original_subtitle[start_idx].start
                end_time = original_subtitle[self.start_line - 2].end
                parts_user = []
                parts_user.append(
                    types.Part(
                        text=json.dumps(
                            [
                                SubtitleObject(
                                    index=str(j),
                                    content=original_subtitle[j].text,
                                    style=original_subtitle[j].style,
                                    name=original_subtitle[j].name,
                                    time_start=str(pysubs2.make_time(msecs=original_subtitle[j].start)) if self.audio_file else None,
                                    time_end=str(pysubs2.make_time(msecs=original_subtitle[j].end)) if self.audio_file else None,
                                )
                                for j in range(start_idx, self.start_line - 1)
                            ],
                            ensure_ascii=False,
                        )
                    )
                )

                parts_model = []
                parts_model.append(
                    types.Part(
                        text=json.dumps(
                            [
                                SubtitleObject(
                                    index=str(j),
                                    content=translated_subtitle[j].text, # Use o texto já traduzido
                                    style=translated_subtitle[j].style, # Manter o estilo
                                    name=translated_subtitle[j].name, # Manter o nome
                                )
                                for j in range(start_idx, self.start_line - 1)
                            ],
                            ensure_ascii=False,
                        )
                    )
                )

                previous_message = [
                    types.Content(
                        role="user",
                        parts=parts_user,
                    ),
                    types.Content(
                        role="model",
                        parts=parts_model,
                    ),
                ]

            highlight(f"Starting translation of {total - self.start_line + 1} lines...\n")
            progress_bar(i - 1, total, prefix="Translating:", suffix=f"{self.model_name}", isSending=True)

            batch.append(
                SubtitleObject(
                    index=str(i),
                    content=original_subtitle[i].text,
                    style=original_subtitle[i].style,
                    name=original_subtitle[i].name,
                    time_start=str(pysubs2.make_time(msecs=original_subtitle[i].start)) if self.audio_file else None,
                    time_end=str(pysubs2.make_time(msecs=original_subtitle[i].end)) if self.audio_file else None,
                )
            )
            i += 1

            if self.gemini_api_key2:
                info_with_progress(f"Starting with API Key {self.current_api_number}")

            def handle_interrupt(signal_received, frame):
                nonlocal translated_file
                last_chunk_size = get_last_chunk_size()
                warning_with_progress(
                    f"Translation interrupted. Saving partial results to file. Progress saved.",
                    chunk_size=max(0, last_chunk_size - 1),
                )
                if translated_file:  # <--- Verifica se o arquivo foi aberto
                    # Substituir a linha original de salvamento pelo bloco abaixo:
                    output_path = translated_file.name #.ass
                    save_format = "ass"
                    # Garantir extensão coerente com o formato escolhido
                    if save_format == "ass" and not translated_file.name.endswith(".ass"):
                        # renomear antes de salvar
                        output_path = translated_file.name.rsplit(".", 1)[0] + ".ass"
                        translated_file.close()

                    translated_subtitle.save(
                        output_path,
                        encoding="utf-8",
                        format=save_format
                    )
                    translated_file.close()
                if self.progress_log:
                    save_logs_to_file(self.log_file_path)
                self._save_progress(max(1, i - len(batch) + max(0, last_chunk_size - 1) + 1))
                exit(0)

            signal.signal(signal.SIGINT, handle_interrupt)

            # Save initial progress
            self._save_progress(i)

            last_time = 0
            validated = False
            while i < total or len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(
                        SubtitleObject(
                            index=str(i),
                            content=original_subtitle[i].text,
                            style=original_subtitle[i].style,
                            name=original_subtitle[i].name,
                            time_start=str(pysubs2.make_time(msecs=original_subtitle[i].start)) if self.audio_file else None,
                            time_end=str(pysubs2.make_time(msecs=original_subtitle[i].end)) if self.audio_file else None,
                        )
                    )
                    i += 1
                    continue
                try:
                    while not validated:
                        info_with_progress(f"Validating token size...")
                        try:
                            validated = self._validate_token_size(json.dumps(batch, ensure_ascii=False))
                        except Exception as e:
                            error_with_progress(f"Error validating token size: {e}")
                            info_with_progress(f"Retrying validation...")
                            continue
                        if not validated:
                            error_with_progress(
                                f"Token size ({int(self.token_count/0.9)}) exceeds limit ({self.token_limit}) for {self.model_name}."
                            )
                            user_prompt = "0"
                            while not user_prompt.isdigit() or int(user_prompt) <= 0:
                                user_prompt = input_prompt_with_progress(
                                    f"Please enter a new batch size (current: {self.batch_size}): ",
                                    batch_size=self.batch_size,
                                )
                                if user_prompt.isdigit() and int(user_prompt) > 0:
                                    new_batch_size = int(user_prompt)
                                    decrement = self.batch_size - new_batch_size
                                    if decrement > 0:
                                        for _ in range(decrement):
                                            i -= 1
                                            batch.pop()
                                    self.batch_size = new_batch_size
                                    info_with_progress(f"Batch size updated to {self.batch_size}.")
                                else:
                                    warning_with_progress("Invalid input. Batch size must be a positive integer.")
                            continue
                        success_with_progress(f"Token size validated. Translating...", isSending=True)

                    if i == total and len(batch) < self.batch_size:
                        self.batch_size = len(batch)

                    start_time = time.time()
                    previous_message = self._process_batch(batch, previous_message, translated_subtitle)
                    end_time = time.time()

                    # Update progress bar
                    progress_bar(i, total, prefix="Translating:", suffix=f"{self.model_name}", isSending=True)

                    # Save progress after each batch
                    self._save_progress(i + 1)

                    if delay and (end_time - start_time < delay_time) and i < total:
                        time.sleep(delay_time - (end_time - start_time))
                except Exception as e:
                    e_str = str(e)
                    last_chunk_size = get_last_chunk_size()

                    if "quota" in e_str:
                        current_time = time.time()
                        if current_time - last_time > 60 and self._switch_api():
                            highlight_with_progress(
                                f"API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...",
                                isSending=True,
                            )
                        else:
                            for j in range(60, 0, -1):
                                warning_with_progress(f"All API quotas exceeded, waiting {j} seconds...")
                                time.sleep(1)
                        last_time = current_time
                    else:
                        i -= self.batch_size
                        j = i + last_chunk_size
                        parts_original = []
                        parts_translated = []
                        for k in range(i, max(i, j)):
                            parts_original.append(
                                SubtitleObject(
                                    index=str(k),
                                    content=original_subtitle[k].text,
                                ),
                            )
                            parts_translated.append(
                                SubtitleObject(index=str(k), content=translated_subtitle[k].text),
                            )
                        if len(parts_translated) != 0:
                            previous_message = [
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=json.dumps(parts_original, ensure_ascii=False))],
                                ),
                                types.Content(
                                    role="model",
                                    parts=[types.Part(text=json.dumps(parts_translated, ensure_ascii=False))],
                                ),
                            ]
                        batch = []
                        progress_bar(
                            i + max(0, last_chunk_size),
                            total,
                            prefix="Translating:",
                            suffix=f"{self.model_name}",
                        )
                        error_with_progress(f"{e_str}")
                        if not self.streaming or last_chunk_size == 0:
                            info_with_progress("Sending last batch again...", isSending=True)
                        else:
                            i += last_chunk_size
                            info_with_progress(f"Resuming from line {i}...", isSending=True)
                        if self.progress_log:
                            save_logs_to_file(self.log_file_path)

            success_with_progress("Translation completed successfully!")
            if self.progress_log:
                save_logs_to_file(self.log_file_path)
            # Substituir a linha original de salvamento pelo bloco abaixo:
            save_format = "ass"
            # Garantir extensão coerente com o formato escolhido
            if save_format == "ass" and not translated_file.name.endswith(".ass"):
                # renomear antes de salvar
                new_name = translated_file.name.rsplit(".", 1)[0] + ".ass"
                translated_file.close()
                translated_file = open(new_name, "w+", encoding="utf-8")

            translated_subtitle.save(
                translated_file.name,
                encoding="utf-8",
                format=save_format
            )
            translated_file.close()

            if self.audio_file and os.path.exists(self.audio_file) and self.audio_extracted:
                os.remove(self.audio_file)

            if self.progress_file and os.path.exists(self.progress_file):
                os.remove(self.progress_file)

        if self.srt_extracted and os.path.exists(self.input_file):
            os.remove(self.input_file)

    def _switch_api(self) -> bool:
        """
        Switch to the secondary API key if available.

        Returns:
            bool: True if switched successfully, False if no alternative API available
        """
        if self.current_api_number == 1 and self.gemini_api_key2:
            self.current_api_key = self.gemini_api_key2
            self.current_api_number = 2
            self.backup_api_number = 1
            return True
        if self.current_api_number == 2 and self.gemini_api_key:
            self.current_api_key = self.gemini_api_key
            self.current_api_number = 1
            self.backup_api_number = 2
            return True
        return False

    def _get_client(self) -> genai.Client:
        """
        Configure and return a Gemini client instance.

        Returns:
            genai.Client: Configured Gemini client instance
        """
        client = genai.Client(api_key=self.current_api_key)
        return client

    def _get_token_limit(self):
        """
        Get the token limit for the current model.

        Returns:
            int: Token limit for the current model
        """
        client = self._get_client()
        model = client.models.get(model=self.model_name)
        self.token_limit = model.output_token_limit

    def _validate_token_size(self, contents: str) -> bool:
        """
        Validate the token size of the input contents.

        Args:
            contents (str): Input contents to validate

        Returns:
            bool: True if token size is valid, False otherwise
        """
        client = self._get_client()
        token_count = client.models.count_tokens(model="gemini-2.0-flash", contents=contents)
        self.token_count = token_count.total_tokens
        if token_count.total_tokens > self.token_limit * 0.9:
            return False
        return True

    def _process_batch(
        self,
        batch: list[SubtitleObject],
        previous_message: list[Content],
        translated_subtitle: list[pysubs2.SSAFile],
    ) -> Content:
        """
        Process a batch of subtitles for translation.

        Args:
            batch (list[SubtitleObject]): Batch of subtitles to translate
            previous_message (Content): Previous message for context
            translated_subtitle (list[pysubs2.SSAFile]): List to store translated subtitles

        Returns:
            Content: The model's response for context in next batch
        """
        client = self._get_client()
        parts = []
        parts.append(types.Part(text=json.dumps(batch, ensure_ascii=False)))
        if self.audio_part:
            parts.append(self.audio_part)

        current_message = types.Content(role="user", parts=parts)
        contents = []
        contents += previous_message
        contents.append(current_message)
        
        self._log_debug(f"--- Processing Batch {self.batch_number} ---")
        self._log_debug(f"Batch size: {len(batch)}")
        self._log_debug(f"Sending {len(contents)} parts to API.")
        self._log_debug(f"Content being sent (last part): {json.dumps(batch, ensure_ascii=False, indent=2)}")

        done = False
        retry = -1
        while done == False:
            response_text = ""
            thoughts_text = ""
            chunk_size = 0
            self.translated_batch = []
            processed = True
            done_thinking = False
            retry += 1
            blocked = False
            self._log_debug(f"Attempt {retry + 1} for batch {self.batch_number}.")
            if not self.streaming:
                response = client.models.generate_content(
                    model=self.model_name, contents=contents, config=self._get_config()
                )
                if response.prompt_feedback:
                    blocked = True
                    break
                if not response.text:
                    error_with_progress("Gemini has returned an empty response.")
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    elif part.thought:
                        thoughts_text += part.text
                        continue
                    else:
                        response_text += part.text
                if self.thoughts_log and self.thinking:
                    if retry == 0:
                        info_with_progress(f"Batch {self.batch_number} thinking process saved to file.")
                    else:
                        info_with_progress(f"Batch {self.batch_number}.{retry} thinking process saved to file.")
                    save_thoughts_to_file(thoughts_text, self.thoughts_file_path, retry)
                
                # Extrai o JSON do texto de resposta antes de tentar o parse
                clean_response_text = extract_json_from_text(response_text)
                self.translated_batch: list[SubtitleObject] = json_repair.loads(clean_response_text)
            else:
                if blocked:
                    break
                response = client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=self._get_config()
                )
                self._log_debug("API stream opened. Waiting for chunks...")
                for chunk in response:
                    if chunk.prompt_feedback:
                        blocked = True
                        break
                    self._log_debug(f"Received chunk: {chunk}")
                    # Processar 'thoughts' primeiro
                    if self.thinking and hasattr(chunk, "thought") and chunk.thought:
                        update_loading_animation(chunk_size=chunk_size, isThinking=True)
                        thoughts_text += chunk.thought
                        continue
                    
                    # Processar texto de conteúdo
                    if chunk.text:
                        if not done_thinking and self.thoughts_log and self.thinking:
                            log_message = f"Batch {self.batch_number}"
                            if retry > 0:
                                log_message += f".{retry}"
                            info_with_progress(f"{log_message} thinking process saved to file.")
                            save_thoughts_to_file(thoughts_text, self.thoughts_file_path, retry)
                            done_thinking = True
                        response_text += chunk.text
                self._log_debug("Finished receiving chunks from API stream.")

                try:
                    # Extrai o JSON do texto de resposta antes de tentar o parse
                    clean_response_text = extract_json_from_text(response_text)
                    if clean_response_text:
                        try:
                            self.translated_batch: list[SubtitleObject] = json_repair.loads(clean_response_text)
                            self._log_debug(f"Successfully parsed JSON from response. Items: {len(self.translated_batch)}")
                        except Exception as e:
                            self._log_debug(f"JSON parse failed. Error: {e}. Raw response: {response_text}")
                            warning_with_progress(f"Could not parse JSON response: {e}")
                            info_with_progress("Sending last batch again...", isSending=True)
                            continue
                    else:
                        warning_with_progress("Received empty response from API.")
                        self._log_debug(f"Received empty or non-JSON response. Raw response: {response_text}")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue # Reinicia o loop para reenviar o lote
                    processed = self._process_translated_lines(
                        translated_lines=self.translated_batch, translated_subtitle=translated_subtitle, batch=batch, finished=False
                    )
                    if not processed:
                        break
                except Exception as e:
                    self._log_debug(f"Exception during stream processing: {e}")
                    warning_with_progress(f"Exception occurred during streaming response processing: {e}")
                    continue
                update_loading_animation(chunk_size=len(self.translated_batch))

            if len(self.translated_batch) == len(batch):
                processed = self._process_translated_lines(
                    translated_lines=self.translated_batch,
                    translated_subtitle=translated_subtitle,
                    batch=batch,
                    finished=True,
                )
                if not processed:
                    info_with_progress("Sending last batch again...", isSending=True)
                    self._log_debug("Processing translated lines failed. Retrying remaining items in batch.")
                    continue
                done = True
                self.batch_number += 1
            else:
                if processed:
                    # This case handles partial success. Some lines were processed.
                    warning_with_progress(
                        f"Gemini has returned an unexpected response. Expected {len(batch)} lines, got {len(self.translated_batch)}."
                    )
                self._log_debug(f"Mismatch in line count. Expected {len(batch)}, got {len(self.translated_batch)}. Retrying.")
                info_with_progress("Sending last batch again...", isSending=True)
                continue

        if blocked:
            error_with_progress(
                "Gemini has blocked the translation for unknown razões. Tente mudar sua descrição (se tiver uma) e/ou o tamanho do lote e tente novamente."
            )
            signal.raise_signal(signal.SIGINT)
        parts = []
        parts.append(types.Part(thought=True, text=thoughts_text)) if thoughts_text else None
        parts.append(types.Part(text=response_text))
        previous_content = [
            types.Content(role="user", parts=[types.Part(text=json.dumps(batch, ensure_ascii=False))]),
            types.Content(role="model", parts=parts),
        ]
        batch.clear()
        return previous_content

    def _process_translated_lines(
        self,
        translated_lines: list[SubtitleObject],
        translated_subtitle: list[pysubs2.SSAFile],
        batch: list[SubtitleObject],
        finished: bool,
    ) -> bool:
        """
        Process translated lines and update the subtitle object.
        If processing is partial, it updates the original batch to contain only unprocessed items.
        """
        if not translated_lines:
            return False

        processed_indexes = set()
        all_successful = True

        for line in translated_lines:
            # Basic validation for the returned object
            if "content" not in line or "index" not in line or not line["index"].isdigit():
                warning_with_progress(f"Gemini returned a malformed object, skipping: {line}")
                all_successful = False
                continue

            line_index = int(line["index"])
            
            # Find the corresponding item in the original batch
            original_item = next((item for item in batch if int(item["index"]) == line_index), None)

            if not original_item:
                warning_with_progress(f"Gemini returned an unexpected index: {line_index}.")
                all_successful = False
                continue

            if self._dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[line_index].text = f"\u202b{line['content']}\u202c"
            else:
                translated_subtitle[line_index].text = line["content"]
            
            processed_indexes.add(str(line_index))

        # If processing was not fully successful, filter the batch to retry only the failed items.
        if not all_successful or len(processed_indexes) < len(batch):
            batch[:] = [item for item in batch if item["index"] not in processed_indexes]
            return False # Indicates partial success, batch has been modified for retry.

        return True # All items in the batch were processed successfully.

    def _dominant_strong_direction(self, s: str) -> str:
        """
        Determine the dominant text direction (RTL or LTR) of a string.

        Args:
            s (str): Input string to analyze

        Returns:
            str: 'rtl' if right-to-left is dominant, 'ltr' otherwise
        """
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count["R"] + count["AL"] + count["RLE"] + count["RLI"]
        ltr_count = count["L"] + count["LRE"] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"


def extract_srt_from_video(video_path, extension=".srt") -> str:
    """
    Extract SRT subtitles from a video file using FFmpeg.
    Returns the path to the extracted SRT file.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    subtitle_path = os.path.join(os.path.dirname(video_path), f"{base_name}_extracted{extension}")
    if os.path.exists(subtitle_path):
        return subtitle_path
    cmd = ["ffmpeg", "-v", "quiet", "-i", video_path, "-map", "0:s:0", "-c:s", extension.strip('.'), subtitle_path]
    try:
        info(f"Extracting subtitles from video file as {extension}...")
        subprocess.run(cmd, check=True, encoding="utf-8")
        return subtitle_path
    except subprocess.CalledProcessError:
        warning(f"FFmpeg command failed to extract {extension}. Trying next format if available.")
        return ""


def check_ffmpeg_installation():
    """
    Check if FFmpeg is installed on the system.
    Returns True if installed, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info("FFmpeg is installed.")
        return True
    except subprocess.CalledProcessError:
        error("FFmpeg is not installed. Please install FFmpeg to use video features.")
        return False
