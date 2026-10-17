import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from openai import OpenAI, APIError

# ==============================================================================
# Configuration Constants
# ==============================================================================

# xAI (Grok) configuration:
API_BASE_URL = "https://api.x.ai/v1"
MODEL_NAME = "grok-beta"

# Alternative: Fast Gemma via Groq:
# API_BASE_URL = "https://api.groq.com/openai/v1"
# MODEL_NAME = "gemma2-9b-it"

DEFAULT_INPUT_FILE = "prompts.txt"
DEFAULT_OUTPUT_FILE = "llm_responses.json"
REQUEST_TIMEOUT_SECONDS = 60.0
TEMPERATURE = 0.7


# ==============================================================================
# Core Functions
# ==============================================================================

def load_prompts_from_file(file_path: Path) -> List[str]:
    """Reads non-empty, stripped lines from a specified text file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    with file_path.open(mode="r", encoding="utf-8") as file:
        prompts = [line.strip() for line in file if line.strip()]

    return prompts


def query_llm(client: OpenAI, prompt: str, model: str) -> Dict[str, Any]:
    """Sends a single prompt to the LLM API and formats the response record."""
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return {
            "prompt": prompt,
            "response": content,
            "status": "success",
            "model": model,
            "timestamp_utc": timestamp,
            "usage": usage,
            "error": None,
        }

    except APIError as api_err:
        return {
            "prompt": prompt,
            "response": None,
            "status": "error",
            "model": model,
            "timestamp_utc": timestamp,
            "usage": None,
            "error": str(api_err),
        }


def save_results_to_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves the collected results list to a formatted JSON file."""
    with output_path.open(mode="w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2, ensure_ascii=False)


def main() -> None:
    """Main execution pipeline."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("Error: Missing 'LLM_API_KEY' environment variable.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(DEFAULT_INPUT_FILE)
    output_path = Path(DEFAULT_OUTPUT_FILE)

    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
    )

    print(f"Loading prompts from '{input_path}'...")
    try:
        prompts = load_prompts_from_file(input_path)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    if not prompts:
        print("No prompts found in the input file.")
        return

    print(f"Found {len(prompts)} prompt(s). Processing with model '{MODEL_NAME}'...\n")

    results: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] Sending: {prompt[:50]}...")
        result = query_llm(client=client, prompt=prompt, model=MODEL_NAME)
        results.append(result)

    save_results_to_json(results, output_path)
    print(f"\nDone. Successfully saved {len(results)} response(s) to '{output_path}'.")


if __name__ == "__main__":
    main()