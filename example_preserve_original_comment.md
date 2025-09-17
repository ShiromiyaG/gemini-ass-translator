# How to use the --preserve-original-comment option

This feature allows you to keep the original text as an invisible comment on a line above the translation in the ASS file.

## CLI Usage

```bash
python -m gemini_srt_translator translate -i input.ass -l "Portuguese" --preserve-original-comment
```

## Result

### Before (original file):
```
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world!
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,How are you?
```

### After (with --preserve-original-comment):
```
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\alpha&HFF&}# Original: Hello world!
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Olá mundo!
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,{\alpha&HFF&}# Original: How are you?
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Como você está?
```

## How it works

1. **Invisible text**: The `{\alpha&HFF&}` tag makes the text completely transparent
2. **Comment**: The `# Original:` prefix identifies the text as a comment
3. **Positioning**: The comment is placed on the line immediately above the translation
4. **Timing**: The comment has the same timing as the translation

## Advantages

- ✅ Preserves original text for reference
- ✅ Does not interfere with visualization (invisible text)
- ✅ Facilitates reviews and corrections
- ✅ Maintains chronological structure
- ✅ Compatible with standard ASS players

## Programmatic Usage

```python
from gemini_srt_translator.main import GeminiSRTTranslator

translator = GeminiSRTTranslator(
    input_file="input.ass",
    target_language="Portuguese",
    preserve_original_as_comment=True  # Enables the feature
)

translator.translate()
```

## Difference from other options

- **Invisible text on same line**: Keeps original and translation on the same line
- **Comment on separate line**: Keeps original on dedicated line (this option)
- **No preservation**: Only the translation (default)