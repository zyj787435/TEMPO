from openai import OpenAI
from ..config import Config
import os
import time

# ---------------------------------------------------------------------------
# Global per-example token accumulator
# Call reset_token_stats() before each example, get_token_stats() after.
# ---------------------------------------------------------------------------
_token_stats = {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}

def reset_token_stats():
    _token_stats["input_tokens"] = 0
    _token_stats["output_tokens"] = 0
    _token_stats["n_calls"] = 0

def get_token_stats():
    return dict(_token_stats)
# ---------------------------------------------------------------------------

def call_model(messages, temperature=None, top_p=None, seed=None):
    # Always use Config.MODEL_NAME from .env, ignore system OPENAI_MODEL_NAME
    model_name = Config.MODEL_NAME

    client = OpenAI(
        base_url=Config.OPENAI_BASE_URL,
        api_key=Config.OPENAI_API_KEY,
    )

    # Build API call parameters — only include supported ones
    api_params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature if temperature is not None else Config.TEMPERATURE,
        top_p=top_p if top_p is not None else Config.TOP_P,
    )
    if Config.STOP is not None:
        api_params['stop'] = Config.STOP

    # frequency_penalty and presence_penalty: only include if non-zero
    if Config.FREQUENCY_PENALTY != 0:
        api_params['frequency_penalty'] = Config.FREQUENCY_PENALTY
    if Config.PRESENCE_PENALTY != 0:
        api_params['presence_penalty'] = Config.PRESENCE_PENALTY

    # seed: some providers (e.g. SiliconFlow) may not support it, skip if issues arise
    effective_seed = seed if seed is not None else Config.SEED
    if effective_seed is not None:
        api_params['seed'] = effective_seed

    # Retry logic with exponential backoff for rate limiting (429) errors
    max_retries = 10
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(**api_params)

            content = completion.choices[0].message.content
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
            completion_tokens = completion.usage.completion_tokens if completion.usage else 0
            _token_stats["input_tokens"] += prompt_tokens
            _token_stats["output_tokens"] += completion_tokens
            _token_stats["n_calls"] += 1
            return content, prompt_tokens, completion_tokens

        except Exception as e:
            error_str = str(e)

            # Handle rate limiting (429) with exponential backoff
            if '429' in error_str or 'rate' in error_str.lower():
                delay = base_delay * (2 ** attempt)
                delay = min(delay, 120)  # cap at 2 minutes
                print(f"[Rate limit] Attempt {attempt+1}/{max_retries}, retrying in {delay}s...")
                time.sleep(delay)
                continue

            # Handle unsupported parameter errors by removing seed and retrying
            # Also catches 400 Bad Request where provider doesn't mention "seed" by name
            if 'seed' in api_params and (
                'seed' in error_str.lower()
                or '400' in error_str
                or 'bad request' in error_str.lower()
            ):
                print(f"[Param error] Removing 'seed' parameter and retrying...")
                del api_params['seed']
                continue

            # For other errors, also retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                delay = min(delay, 60)
                print(f"[API error] Attempt {attempt+1}/{max_retries}: {error_str[:200]}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                raise

    raise RuntimeError(f"LLM call failed after {max_retries} retries")


if __name__ == '__main__':
    messages = [{'role': 'user', 'content': 'Test'}]
    print(call_model(messages))
