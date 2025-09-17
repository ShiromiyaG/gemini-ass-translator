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
import re   # <--- ADD

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
    success,
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
    import re
    
    # Remove markdown fences and other non-JSON text
    text = text.strip()
    
    # Remove common markdown code block patterns
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove any leading/trailing non-JSON text
    text = re.sub(r'^[^{\[]*', '', text)
    text = re.sub(r'[^}\]]*$', '', text)
    
    # Find JSON array or object patterns with better matching
    json_patterns = [
        r'\[(?:[^[\]]*(?:\[(?:[^[\]]*(?:\[[^[\]]*\])*[^[\]]*)*\])*[^[\]]*)*\]',  # Nested array pattern
        r'\{(?:[^{}]*(?:\{(?:[^{}]*(?:\{[^{}]*\})*[^{}]*)*\})*[^{}]*)*\}',  # Nested object pattern
        r'\[.*?\]',  # Simple array pattern
        r'\{.*?\}',  # Simple object pattern
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the largest match (most likely to be complete JSON)
            largest_match = max(matches, key=len)
            # Basic validation - ensure it starts and ends correctly
            if ((largest_match.startswith('[') and largest_match.endswith(']')) or 
                (largest_match.startswith('{') and largest_match.endswith('}'))):
                return largest_match
    
    # If no pattern matches, try to find start and end manually with bracket counting
    start_chars = ['[', '{']
    
    for start_char in start_chars:
        start_pos = text.find(start_char)
        if start_pos != -1:
            end_char = ']' if start_char == '[' else '}'
            bracket_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[start_pos:], start_pos):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            return text[start_pos:i + 1]
    
    # Return original text if no JSON structure is found
    return text


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
        source_language: str = None,  # <--- novo parâmetro
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
        preserve_format: bool = True,
        preserve_original_as_comment: bool = False,  # <--- novo parâmetro
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
            preserve_original_as_comment (bool): Whether to add original text as invisible comment above translation
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

        self.progress_file = os.path.join(dir_path, f"{base_name}.progress") if dir_path else f"{base_name}.progress"

        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.target_language = target_language
        self.source_language = source_language
        self.input_file = input_file
        self.output_file = output_file  # Será redefinido depois se necessário
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
        self.preserve_original_as_comment = preserve_original_as_comment
        # Inicializar o dicionário para armazenar textos originais
        self._original_texts_for_comments = {}
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

        self.translation_cache: dict[str, str] = {}  # cache texto visível original -> tradução
        self._pending_intra_batch_duplicates: list[dict] = []  # duplicadas aguardando propagação dentro do batch
        self.skipped_duplicates = 0  # contador global de linhas reaproveitadas

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
                source_language=self.source_language,  # <--- adicionar esta linha
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
        if self.start_line is not None:
            return
        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                saved_line = data.get("line", 1)          # 1-based salvo
                input_file = data.get("input_file")
                if input_file != self.input_file:
                    warning(f"Found progress file for different subtitle: {input_file}")
                    warning("Ignoring saved progress.")
                    return
                if saved_line > 1:
                    if self.resume is None:
                        resume = input_prompt("Found saved progress. Resume? (y/n): ", mode="resume").lower().strip()
                    elif self.resume is True:
                        resume = "y"
                    else:
                        resume = "n"
                    if resume in ("y", "yes"):
                        info(f"Resuming from line {saved_line}")
                        # converter para 0-based interno
                        self.start_line = saved_line - 1
                    else:
                        info("Starting from the beginning")
                        try:
                            os.remove(self.progress_file)  # Corrigido: remover progress, não o arquivo traduzido
                        except:
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

    # --- NEW HELPERS ---
    def _split_leading_tags(self, text: str) -> tuple[str, str]:
        r"""
        Separa tags iniciais {..}{..} do restante (texto visível).
        Preserva \\N no texto visível.
        """
        if not text:
            return "", ""
        m = re.match(r'^((?:\{[^}]*\})+)', text)
        if m:
            return m.group(1), text[m.end():]
        return "", text

    def _reconstruct_with_tags(self, tags: str, translated_visible: str) -> str:
        return f"{tags}{translated_visible}"

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

        # Gerar output_file padrão mantendo a extensão original se não foi especificado
        if not self.output_file:
            base_name, _ext = os.path.splitext(self.input_file)
            # Preservar a extensão original por padrão
            if self.original_ext:
                extension = self.original_ext
            else:
                extension = ".ass"
            self.output_file = f"{base_name}_translated{extension}"

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
            if not hasattr(self, "translation_cache"):
                self.translation_cache = {}
            # batch_visible_map controla textos visíveis únicos dentro do batch atual
            batch_visible_map: dict[str, int] = {}
            if not original_subtitle:
                error("Subtitle file is empty.")
                return
            try:
                translated_file_exists = open(self.output_file, "r", encoding="utf-8")
                translated_subtitle = pysubs2.load(translated_file_exists.read())
                info(f"Translated file {self.output_file} already exists. Loading existing translation...\n")
                if self.start_line is None:
                    # pedir linha (1-based) só se não veio de resume
                    while True:
                        try:
                            user_line = int(
                                input_prompt(
                                    f"Enter the line number to start from (1 to {len(original_subtitle)}): ",
                                    mode="line",
                                    max_length=len(original_subtitle),
                                ).strip()
                            )
                            if 1 <= user_line <= len(original_subtitle):
                                self.start_line = user_line - 1  # armazenar 0-based
                                break
                            else:
                                warning(f"Line number must be between 1 and {len(original_subtitle)}. Please try again.")
                        except ValueError:
                            warning("Invalid input. Please enter a valid number.")
            except FileNotFoundError:
                # Create a new copy of the original subtitle for translation
                translated_subtitle = copy.deepcopy(original_subtitle)
                self.start_line = 0
    
            if len(original_subtitle) != len(translated_subtitle):
                error("Number of lines of existing translated file does not match the number of lines in the original file.", ignore_quiet=True)
                exit(1)

            # Adiar abertura até o final; remover linha antiga:
            # translated_file = open(self.output_file, "w", encoding="utf-8")
            translated_file = None

            if not (0 <= self.start_line < len(original_subtitle)):
                error(f"Start line must be between 1 and {len(original_subtitle)}.", ignore_quiet=True)
                exit(1)

            if len(original_subtitle) < self.batch_size:
                self.batch_size = len(original_subtitle)

            i = self.start_line  # 0-based próxima linha a processar
            batch = []
            previous_message = []
            if self.start_line > 0:
                start_idx = max(0, self.start_line - self.batch_size)
                parts_user = [
                    types.Part(
                        text=json.dumps(
                            [
                                SubtitleObject(
                                    index=str(j + 1),  # 1-based para o modelo
                                    content=original_subtitle[j].text,
                                    style=original_subtitle[j].style,
                                    name=original_subtitle[j].name,
                                    time_start=str(pysubs2.make_time(msecs=original_subtitle[j].start)) if self.audio_file else None,
                                    time_end=str(pysubs2.make_time(msecs=original_subtitle[j].end)) if self.audio_file else None,
                                )
                                for j in range(start_idx, self.start_line)
                            ],
                            ensure_ascii=False,
                        )
                    )
                ]
                parts_model = [
                    types.Part(
                        text=json.dumps(
                            [
                                SubtitleObject(
                                    index=str(j + 1),
                                    content=translated_subtitle[j].text,
                                    style=translated_subtitle[j].style,
                                    name=translated_subtitle[j].name,
                                )
                                for j in range(start_idx, self.start_line)
                            ],
                            ensure_ascii=False,
                        )
                    )
                ]
                previous_message = [
                    types.Content(role="user", parts=parts_user),
                    types.Content(role="model", parts=parts_model),
                ]

            remaining = total - i
            highlight(f"Starting translation of {remaining} lines...\n")
            progress_bar(i, total, prefix="Translating:", suffix=f"{self.model_name}", isSending=True)

            if i < total:
                # --- DUP CHECK (primeira linha do lote) ---
                tags_prefix, visible_part = self._split_leading_tags(original_subtitle[i].text)
                key = visible_part
                if key and key in self.translation_cache:
                    # Reusar tradução
                    translated_visible = self.translation_cache[key]
                    translated_subtitle[i].text = self._reconstruct_with_tags(tags_prefix, translated_visible)
                    # Armazenar texto original para comentário posterior (linhas com cache)
                    if self.preserve_original_as_comment:
                        self._original_texts_for_comments[i] = original_subtitle[i].text
                    self.skipped_duplicates += 1
                    i += 1
                else:
                    if key and key in batch_visible_map:
                        # duplicada dentro do mesmo batch (primeira ainda não traduzida)
                        self._pending_intra_batch_duplicates.append({
                            "dup_index": i,
                            "orig_index": batch_visible_map[key],
                            "dup_tags": tags_prefix,
                            "dup_visible": key,
                            "original_text": original_subtitle[i].text,  # Adicionar texto original
                        })
                        i += 1
                    else:
                        if key:
                            batch_visible_map[key] = i
                        batch.append(
                            SubtitleObject(
                                index=str(i + 1),
                                content=original_subtitle[i].text,
                                style=original_subtitle[i].style,
                                name=original_subtitle[i].name,
                                time_start=str(pysubs2.make_time(msecs=original_subtitle[i].start)) if self.audio_file else None,
                                time_end=str(pysubs2.make_time(msecs=original_subtitle[i].end)) if self.audio_file else None,
                            )
                        )
                        i += 1

            # salvar progresso: próximo (1-based)
            self._save_progress(i + 1)

            # ...no loop principal substituir condição e appends...
            # Inicializações de controle para o loop principal
            last_time = 0.0  # usado em tratamento de quota
            min_interval = 0.0
            if self.free_quota:
                # Se estiver usando quota gratuita, ser mais conservador entre lotes maiores
                # Ajuste simples: 0.5s para lotes até 50, 1s para lotes maiores
                if self.batch_size > 200:
                    min_interval = 1.0
                elif self.batch_size > 50:
                    min_interval = 0.5
                else:
                    min_interval = 0.2
            else:
                # Usuário com quota paga pode enviar mais rápido
                if self.batch_size > 200:
                    min_interval = 0.4
                elif self.batch_size > 50:
                    min_interval = 0.15
                else:
                    min_interval = 0.05

            while i < total or len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    # --- DUP CHECK (demais linhas) ---
                    tags_prefix, visible_part = self._split_leading_tags(original_subtitle[i].text)
                    key = visible_part
                    if key and key in self.translation_cache:
                        translated_visible = self.translation_cache[key]
                        translated_subtitle[i].text = self._reconstruct_with_tags(tags_prefix, translated_visible)
                        # Armazenar texto original para comentário posterior (linhas duplicadas)
                        if self.preserve_original_as_comment:
                            self._original_texts_for_comments[i] = original_subtitle[i].text
                        self.skipped_duplicates += 1
                        i += 1
                        continue
                    if key and key in batch_visible_map:
                        self._pending_intra_batch_duplicates.append({
                            "dup_index": i,
                            "orig_index": batch_visible_map[key],
                            "dup_tags": tags_prefix,
                            "dup_visible": key,
                            "original_text": original_subtitle[i].text,  # Adicionar texto original
                        })
                        i += 1
                        continue
                    if key:
                        batch_visible_map[key] = i
                    batch.append(
                        SubtitleObject(
                            index=str(i + 1),
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
                    # Garantir que validated esteja definido a cada iteração de envio
                    validated = False
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
                    # Controle de taxa substituindo bloco antigo com variáveis indefinidas (delay/delay_time)
                    # Garante um intervalo mínimo entre lotes para evitar limites de API.
                    elapsed = end_time - start_time
                    if i < total and elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
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
            if self.skipped_duplicates:
                info(f"Skipped {self.skipped_duplicates} duplicate lines (cache reuse + intra-batch).")
            if self.progress_log:
                save_logs_to_file(self.log_file_path)

            # Adicionar comentários com texto original se habilitado
            if hasattr(self, 'preserve_original_as_comment') and self.preserve_original_as_comment and hasattr(self, '_original_texts_for_comments'):
                info_with_progress("Adding original text as comments...")
                self._log_debug(f"preserve_original_as_comment = {self.preserve_original_as_comment}")
                self._log_debug(f"_original_texts_for_comments has {len(self._original_texts_for_comments)} entries")
                self._add_original_comments(translated_subtitle)
                success_with_progress("Original text comments added successfully!")
            elif hasattr(self, 'preserve_original_as_comment') and self.preserve_original_as_comment:
                warning_with_progress("preserve_original_as_comment is enabled but no original texts were stored")
            
            # Salvamento simplificado
            translated_subtitle.save(self.output_file, encoding="utf-8", format="ass")
            success(f"Translation saved to: {self.output_file}")

            # Cleanup files
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
                
                # Processa a resposta com tratamento de erro melhorado
                try:
                    clean_response_text = extract_json_from_text(response_text)
                    self._log_debug(f"Extracted JSON: {clean_response_text}")
                    
                    if not clean_response_text or clean_response_text.strip() in ['', '{}', '[]']:
                        self._log_debug("Empty or invalid JSON extracted from response")
                        warning_with_progress("Received empty JSON response from API.")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue
                    
                    # Tenta fazer o parse do JSON
                    self.translated_batch: list[SubtitleObject] = json_repair.loads(clean_response_text)
                    self._log_debug(f"Successfully parsed JSON. Items: {len(self.translated_batch)}")
                    
                    # Verifica se o resultado é uma lista válida
                    if not isinstance(self.translated_batch, list):
                        self._log_debug(f"JSON is not a list. Type: {type(self.translated_batch)}")
                        warning_with_progress("API returned invalid format (not a list).")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue
                        
                except json.JSONDecodeError as e:
                    self._log_debug(f"JSON decode error: {e}. Raw response: {response_text}")
                    warning_with_progress(f"JSON decode error: {e}")
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue
                except Exception as e:
                    self._log_debug(f"Unexpected error during JSON processing: {e}. Raw response: {response_text}")
                    warning_with_progress(f"Error processing response: {e}")
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue
                    
            else:
                # Streaming processing with improved error handling
                if blocked:
                    break
                response = client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=self._get_config()
                )
                self._log_debug("API stream opened. Waiting for chunks...")
                
                try:
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
                            
                except Exception as e:
                    self._log_debug(f"Error during streaming: {e}")
                    warning_with_progress(f"Streaming error: {e}")
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue

                self._log_debug("Finished receiving chunks from API stream.")

                # Process the complete response
                try:
                    clean_response_text = extract_json_from_text(response_text)
                    self._log_debug(f"Extracted JSON from stream: {clean_response_text}")
                    
                    if not clean_response_text or clean_response_text.strip() in ['', '{}', '[]']:
                        self._log_debug("Empty or invalid JSON extracted from streaming response")
                        warning_with_progress("Received empty JSON response from streaming API.")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue
                    
                    # Try multiple JSON parsing strategies
                    parsed_json = None
                    parsing_error = None
                    
                    # First attempt: json_repair.loads
                    try:
                        parsed_json = json_repair.loads(clean_response_text)
                        self._log_debug("Successfully parsed JSON using json_repair.loads")
                    except Exception as e:
                        parsing_error = e
                        self._log_debug(f"json_repair.loads failed: {e}")
                        
                        # Second attempt: standard json.loads
                        try:
                            parsed_json = json.loads(clean_response_text)
                            self._log_debug("Successfully parsed JSON using standard json.loads")
                        except Exception as e2:
                            self._log_debug(f"Standard json.loads also failed: {e2}")
                            
                            # Third attempt: Try to clean up common issues
                            try:
                                # Remove any trailing commas and fix common issues
                                cleaned_text = re.sub(r',\s*}', '}', clean_response_text)
                                cleaned_text = re.sub(r',\s*]', ']', cleaned_text)
                                parsed_json = json.loads(cleaned_text)
                                self._log_debug("Successfully parsed JSON after manual cleanup")
                            except Exception as e3:
                                self._log_debug(f"Manual cleanup JSON parsing also failed: {e3}")
                    
                    if parsed_json is None:
                        self._log_debug(f"All JSON parsing attempts failed. Raw response: {response_text[:500]}...")
                        warning_with_progress(f"Streaming error: {parsing_error}")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue
                    
                    self.translated_batch: list[SubtitleObject] = parsed_json
                    self._log_debug(f"Successfully parsed streaming JSON. Items: {len(self.translated_batch)}")
                    
                    if not isinstance(self.translated_batch, list):
                        self._log_debug(f"Streaming JSON is not a list. Type: {type(self.translated_batch)}")
                        warning_with_progress("Streaming API returned invalid format (not a list).")
                        info_with_progress("Sending last batch again...", isSending=True)
                        continue
                        
                except Exception as e:
                    self._log_debug(f"Unexpected streaming error: {e}. Raw response: {response_text[:500]}...")
                    warning_with_progress(f"Streaming processing error: {e}")
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue

            # Process translated lines if we got valid data
            if len(self.translated_batch) == len(batch):
                processed = self._process_translated_lines(
                    translated_lines=self.translated_batch,
                    translated_subtitle=translated_subtitle,
                    batch=batch,
                    finished=True,
                )
                if not processed:
                    info_with_progress("Sending last batch again...", isSending=True)
                    continue
                done = True
                self.batch_number += 1
            else:
                if len(self.translated_batch) > 0:
                    warning_with_progress(
                        f"Expected {len(batch)} lines, got {len(self.translated_batch)}. Retrying batch."
                    )
                else:
                    warning_with_progress("No valid translations received. Retrying batch.")
                self._log_debug(f"Mismatch in line count. Expected {len(batch)}, got {len(self.translated_batch)}.")
                info_with_progress("Sending last batch again...", isSending=True)
                continue

        if blocked:
            error_with_progress(
                "Gemini has blocked the translation for unknown reasons. Try changing your description (if you have one) and/or the batch size and try again."
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
        """Process and apply translated lines to subtitle file"""
        all_successful = True
        processed_indexes = set()

        for line in translated_lines:
            try:
                line_index_1b = int(line["index"])
                line_index = line_index_1b - 1
            except (ValueError, KeyError):
                warning_with_progress(f"Invalid index: {line.get('index', 'unknown')}")
                all_successful = False
                continue

            original_item = next((item for item in batch if int(item["index"]) == line_index_1b), None)
            if not (0 <= line_index < len(translated_subtitle)):
                warning_with_progress(f"Index out of range: {line_index_1b}")
                all_successful = False
                continue

            if self._dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[line_index].text = f"\u202b{line['content']}\u202c"
            else:
                translated_subtitle[line_index].text = line["content"]
            
            # Armazenar texto original para comentário posterior
            if original_item and self.preserve_original_as_comment:
                self._original_texts_for_comments[line_index] = original_item["content"]
                self._log_debug(f"Stored original text for line {line_index}: {original_item['content'][:50]}...")

            # Cache store
            if original_item:
                _, orig_visible = self._split_leading_tags(original_item["content"])
                tags_tr, trans_visible = self._split_leading_tags(translated_subtitle[line_index].text)
                if orig_visible and trans_visible:
                    self.translation_cache.setdefault(orig_visible, trans_visible)
            processed_indexes.add(str(line_index))

        # Processar duplicatas pendentes intra-batch
        if self._pending_intra_batch_duplicates:
            self._log_debug(f"Processing {len(self._pending_intra_batch_duplicates)} pending intra-batch duplicates")
            for dup_info in self._pending_intra_batch_duplicates:
                dup_index = dup_info["dup_index"]
                orig_index = dup_info["orig_index"]
                dup_tags = dup_info["dup_tags"]
                
                # Copiar tradução da linha original
                if orig_index < len(translated_subtitle) and dup_index < len(translated_subtitle):
                    _, trans_visible = self._split_leading_tags(translated_subtitle[orig_index].text)
                    translated_subtitle[dup_index].text = self._reconstruct_with_tags(dup_tags, trans_visible)
                    
                    # Armazenar texto original para comentário se disponível
                    if self.preserve_original_as_comment and "original_text" in dup_info:
                        self._original_texts_for_comments[dup_index] = dup_info["original_text"]
                        self._log_debug(f"Stored original text for duplicate line {dup_index}")
                    
                    self.skipped_duplicates += 1
            
            # Limpar lista de duplicatas pendentes
            self._pending_intra_batch_duplicates.clear()

        if not all_successful or len(processed_indexes) < len(batch):
            return False
        return True

    def _add_original_comments(self, translated_subtitle: pysubs2.SSAFile):
        """
        Adiciona comentários invisíveis com texto original acima das traduções.
        """
        if not self._original_texts_for_comments:
            self._log_debug("No original texts stored for comments")
            return
        
        self._log_debug(f"Adding original comments for {len(self._original_texts_for_comments)} lines")
        
        # Criar lista de novos eventos ordenados por índice
        new_events = []
        
        # Processar eventos em ordem, inserindo comentários quando necessário
        for i, event in enumerate(translated_subtitle.events):
            # Se temos texto original para este índice, adicionar comentário antes
            if i in self._original_texts_for_comments:
                original_text = self._original_texts_for_comments[i]
                
                # Limpar o texto original de tags ASS para o comentário
                clean_original = self._clean_ass_tags(original_text)
                
                self._log_debug(f"Creating comment for line {i}: {clean_original[:50]}...")
                
                # Criar evento de comentário invisível
                comment_event = pysubs2.SSAEvent()
                comment_event.start = event.start
                comment_event.end = event.end
                # Usar tag de transparência total para tornar invisível
                comment_event.text = f"{{\\alpha&HFF&}}# Original: {clean_original}"
                comment_event.style = event.style
                comment_event.name = event.name
                comment_event.marginl = event.marginl
                comment_event.marginr = event.marginr
                comment_event.marginv = event.marginv
                comment_event.effect = event.effect
                comment_event.layer = event.layer
                
                new_events.append(comment_event)
            
            # Adicionar o evento original (tradução)
            new_events.append(event)
        
        # Substituir os eventos na legenda
        original_event_count = len(translated_subtitle.events)
        translated_subtitle.events = new_events
        self._log_debug(f"Added {len(new_events) - original_event_count} comment events")

    def _clean_ass_tags(self, text: str) -> str:
        """
        Remove tags ASS do texto para usar em comentários.
        """
        import re
        # Remove tags ASS como {\tag} e {\tag&value&}
        cleaned = re.sub(r'\{[^}]*\}', '', text)
        return cleaned.strip()

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
