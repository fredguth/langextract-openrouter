        # LangExtract openrouter Provider

A provider plugin for LangExtract that supports openrouter models.

## Installation

From source:
```bash
pip install -e .
```

From PyPI:
```bash
pip install langextract-openrouter
```

## Supported Model IDs

- `openrouter*`: Models matching pattern ^openrouter
- `openrouter/google/gemini-2.5-flash`
- `openrouter/openai/gpt-4.1`
- ...

## Environment Variables

- `OPENROUTER_API_KEY`: API key for authentication

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="openrouter/openai/gpt-4.1", # for example
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
