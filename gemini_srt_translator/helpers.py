from google.genai import types
import os
import sys
import typing


def get_instruction(
    language: str, thinking: bool, thinking_compatible: bool, audio_file: str = None, description: str = None
) -> str:
    """
    Get the instruction for the translation model based on the target language.
    """

    thinking_instruction = (
        "\nThink deeply and reason as much as possible before returning the response."
        if thinking
        else "\nDo NOT think or reason."
    )

    fields = (
        "- index: a string identifier\n"
        "- content: the text to translate\n"
        "- style: the style of the subtitle\n"
        "- name: the name of the speaker\n"
        + ("- time_start: the start time of the segment\n- time_end: the end time of the segment\n" if audio_file else "")
    )

    instruction = (
        f"You are an assistant that translates subtitles from any language to {language}.\n"
        f"You will receive a list of objects, each with these fields:\n\n"
        f"{fields}"
        f"\nTranslate the 'content' field of each object.\n"
        f"If the 'content' field is empty, leave it as is.\n"
        f"Preserve line breaks, formatting, and special characters.\n"
        f"Do NOT move or merge 'content' between objects.\n"
        f"Do NOT add or remove any objects.\n"
        f"Do NOT alter the 'index' field.\n"
        f"\n**CRITICAL INSTRUCTIONS FOR .ASS SUBTITLES:**\n"
        f"1.  **NEVER** translate text inside curly braces `{{...}}`. These are formatting tags. Keep them IDENTICAL.\n"
        f"    - Example: `{{\\an8\\c&H0000FF&}}` must remain `{{\\an8\\c&H0000FF&}}`.\n"
        f"2.  The `\\N` characters are for line breaks. Preserve them exactly where they are.\n"
        f"    - Example: `Hello\\NWorld` should be translated as `Olá\\NMundo`.\n"
        f"3.  Only translate the plain text outside of the `{{...}}` tags.\n"
        f"    - Example Input: `{{\\an8}}This is a red text.\\NAnd this is a new line.`\n"
        f"    - Example Correct Output for {language}: `{{\\an8}}Este é um texto vermelho.\\NE esta é uma nova linha.`\n"
    )

    if audio_file:
        instruction += (
            f"\nYou will also receive an audio file.\n"
            f"Use the time_start and time_end of each object to analyze the audio.\n"
            f"Analyze the speaker's voice in the audio to determine gender, then apply grammatical gender rules for {language}:\n"
            f"1. Listen for voice characteristics to identify if speaker is male/female:\n"
            f"   - Use masculine verb forms/adjectives if speaker sounds male\n"
            f"   - Use feminine verb forms/adjectives if speaker sounds female\n"
            f"   - Apply gender agreement to: verbs, adjectives, past participles, pronouns\n"
            f"   - Example: French 'I am tired' -> 'Je suis fatigué' (male) vs 'Je suis fatiguée' (female)\n"
            f"2. In some cases you also need to identify who the current speaker is talking to:\n"
            f"   - If the speaker is talking to a male, use masculine forms.\n"
            f"   - If the speaker is talking to a female, use feminine forms.\n"
            f"   - If the speaker is talking to a group, use plural forms.\n"
            f"   - Example: Portuguese 'You are tired' -> 'Você está cansado' (male) vs 'Você está cansada' (female)\n"
            f"   - Example: Spanish 'You are tired' (group) -> 'Ustedes están cansados' (male/general group) vs 'Ustedes están cansadas' (female group)\n"
        )

    if thinking_compatible:
        instruction += thinking_instruction

    if description:
        instruction += f"\n\nAdditional user instruction:\n\n{description}"

    return instruction


def process_subtitle_file(input_file: str, output_file: str = None) -> typing.Tuple[str, str]:
    """
    Process the subtitle file, checking for its existence and generating an output file path if not provided.
    """
    if not os.path.isfile(input_file):
        print(f"Subtitle file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    if not output_file:
        input_dir = os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        base_name, original_ext = os.path.splitext(input_filename)
        output_file = os.path.join(input_dir, f"{base_name}_translated{original_ext}")

    return input_file, output_file


def get_safety_settings(free_quota: bool = True) -> list[types.SafetySetting]:
    """
    Get the safety settings for the translation model.
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


def get_response_schema() -> types.Schema:
    """
    Get the response schema for the translation model.
    """
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "index": types.Schema(type="STRING"),
                "content": types.Schema(type="STRING"),
            },
            required=["index", "content"],
        ),
    )


if __name__ == "__main__":
    ### Example usage of the get_instruction function

    result = get_instruction(
        language="French",
        description="Translate the subtitles accurately.",
        thinking=True,
        audio_file="example_audio.mp3",
        thinking_compatible=True,
    )
    print(result)
