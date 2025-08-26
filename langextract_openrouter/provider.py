"""Provider implementation for openrouter."""
import logging
import os
import langextract as lx
from openai import OpenAI

_OPENROUTER_ALLOWED = {
    "stop",                 # str | list[str]
    "stream",               # bool
    "max_tokens",           # int
    "temperature",          # float
    "top_p",                # float
    "seed",                 # int
    "top_k",                # int
    "frequency_penalty",    # float
    "presence_penalty",     # float
    "repetition_penalty",   # float
    "min_p",                # float
    "top_a",                # float
    "logit_bias",           # dict
    "top_logprobs",         # int
    "user",                 # str
}

def sanitize_openrouter_kwargs(kwargs: dict) -> dict:
    """
    Filter kwargs to only those OpenRouter supports.
    Logs and drops anything else.
    """
    clean = {}
    for k, v in kwargs.items():
        if k in _OPENROUTER_ALLOWED:
            clean[k] = v
        else:
            logging.debug(f"[openrouter] Dropping unsupported param: {k}={v!r}")
    return clean


@lx.providers.registry.register(r'^openrouter', priority=10)
class openrouterLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for openrouter.

    This provider handles model IDs matching: ['^openrouter']
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the openrouter provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        if model_id.startswith("openrouter/"):
            self.model_id = model_id[11:]  # Remove 'openrouter/' prefix
        else:
            self.model_id = model_id
        self.original_model_id = model_id
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')

        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        self._extra_kwargs = kwargs
        logging.info(f'Initialized OpenRouter provider for model: {self.model_id}')

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        # Merge kwargs with extra_kwargs

        for prompt in batch_prompts:
            try:
                logging.info(f'Calling OpenRouter completion for model {self.model_id}')
                call_params = sanitize_openrouter_kwargs({**self._extra_kwargs, **kwargs})
                result = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": str(prompt)}],
                    response_format={"type": "json_object"},
                    **call_params
                )
                yield [lx.inference.ScoredOutput(score=1.0, output=result.choices[0].message.content)]
            except Exception as e:
                logging.error(f'Error calling OpenRouter completion for model {self.model_id}: {e}')
                yield [lx.inference.ScoredOutput(score=0.0, output='')]  # Return empty output on error
