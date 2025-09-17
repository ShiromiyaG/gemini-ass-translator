"""
# Gemini SRT Translator
    A tool to translate subtitles using Google Generative AI.

## Usage:

### Translate Subtitles
    You can translate subtitles using the `translate` command:
    ```
    import gemini_srt_translator as gst

    gst.gemini_api_key = "your_gemini_api_key_here"
    gst.target_language = "French"
    gst.input_file = "subtitle.srt"

    gst.translate()
    ```
    This will translate the subtitles in the `subtitle.srt` file to French.

### List Models
    You can list the available models using the `listmodels` command:
    ```
    import gemini_srt_translator as gst

    gst.gemini_api_key = "your_gemini_api_key_here"
    gst.listmodels()
    ```
    This will print a list of available models to the console.

"""

import os

from .logger import set_quiet_mode
from .main import GeminiSRTTranslator
from .utils import upgrade_package

gemini_api_key: str = os.getenv("GEMINI_API_KEY", None)
gemini_api_key2: str = os.getenv("GEMINI_API_KEY2", None)
target_language: str = None
source_language: str = None  # <--- adicionar esta linha
input_file: str = None
output_file: str = None
video_file: str = None
audio_file: str = None
extract_audio: bool = None
start_line: int = None
description: str = None
model_name: str = None
batch_size: int = None
streaming: bool = None
thinking: bool = None
thinking_budget: int = None
temperature: float = None
top_p: float = None
top_k: int = None
free_quota: bool = None
skip_upgrade: bool = None
use_colors: bool = True
progress_log: bool = None
thoughts_log: bool = None
quiet: bool = None
resume: bool = None
debug: bool = None
preserve_original_as_comment: bool = None


def getmodels():
    """
    ## Retrieves available models from the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and retrieves a list of available models.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    models = gst._getmodels()
    print(models)
    ```

    Raises:
        Exception: If the Gemini API key is not provided.
    """
    translator = GeminiSRTTranslator(gemini_api_key=gemini_api_key)
    return translator.getmodels()


def listmodels():
    """
    ## Lists available models from the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and retrieves a list of available models. It then prints each model to the console.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    gst.listmodels()
    ```

    Raises:
        Exception: If the Gemini API key is not provided.
    """
    translator = GeminiSRTTranslator(gemini_api_key=gemini_api_key)
    models = translator.getmodels()
    if models:
        print("Available models:\n")
        for model in models:
            print(model)
    else:
        print("No models available or an error occurred while fetching models.")


def translate():
    """
    ## Translates a subtitle file using the Gemini API.
    """
    params = {
        "gemini_api_key": gemini_api_key,
        "gemini_api_key2": gemini_api_key2,
        "target_language": target_language,
        "source_language": source_language,
        "input_file": input_file,
        "output_file": output_file,
        "video_file": video_file,
        "audio_file": audio_file,
        "extract_audio": extract_audio,
        "start_line": start_line,
        "description": description,
        "model_name": model_name,
        "batch_size": batch_size,
        "streaming": streaming,
        "thinking": thinking,
        "thinking_budget": thinking_budget,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "free_quota": free_quota,
        "use_colors": use_colors,
        "progress_log": progress_log,
        "thoughts_log": thoughts_log,
        "resume": resume,
        "debug": debug,
        "preserve_original_as_comment": preserve_original_as_comment,
    }

    if not skip_upgrade:
        try:
            upgrade_package("gemini-srt-translator", use_colors=use_colors)
            raise Exception("Upgrade completed.")
        except Exception:
            pass

    if quiet:
        set_quiet_mode(quiet)

    # Filter out None values
    filtered_params = {k: v for k, v in params.items() if v is not None}
    translator = GeminiSRTTranslator(**filtered_params)
    translator.translate()
