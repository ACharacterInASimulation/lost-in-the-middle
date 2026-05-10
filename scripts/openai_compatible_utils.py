#!/usr/bin/env python3
"""Utilities for calling OpenAI-compatible local model servers."""
import json
import logging
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def parse_extra_body(extra_body_json):
    if not extra_body_json:
        return {}
    try:
        extra_body = json.loads(extra_body_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse --extra-body-json as JSON: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    return extra_body


def clean_model_answer(text, strip_thinking):
    if text is None:
        return ""
    text = text.strip()
    if strip_thinking:
        text = THINKING_RE.sub("", text).strip()
    return text


def complete_prompt(
    prompt,
    api_base,
    api_key,
    model,
    endpoint,
    temperature,
    top_p,
    max_new_tokens,
    system_prompt,
    extra_body,
    stop,
    timeout,
    max_retries,
    retry_sleep,
):
    api_base = api_base.rstrip("/")
    endpoint_path = "chat/completions" if endpoint == "chat" else "completions"
    path = f"/{endpoint_path}" if api_base.endswith("/v1") else f"/v1/{endpoint_path}"
    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }
    if stop:
        payload["stop"] = stop
    if endpoint == "chat":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload["messages"] = messages
    else:
        payload["prompt"] = prompt
    payload.update(extra_body)

    response = post_json(
        url=f"{api_base}{path}",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    return extract_text(response, endpoint)


def post_json(url, payload, api_key, timeout, max_retries, retry_sleep):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt >= max_retries or not should_retry:
                raise RuntimeError(f"Request to {url} failed with HTTP {exc.code}: {error_body}") from exc
            logger.warning("Request failed with HTTP %s; retrying in %.1fs", exc.code, retry_sleep)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Request to {url} failed: {exc}") from exc
            logger.warning("Request failed: %s; retrying in %.1fs", exc, retry_sleep)
        time.sleep(retry_sleep)

    raise RuntimeError(f"Request to {url} failed unexpectedly")


def extract_text(response, endpoint):
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected response shape: {response}") from exc

    if endpoint == "chat":
        message = choice.get("message", {})
        content = message.get("content")
        if content is None:
            content = choice.get("text", "")
        return content or ""
    return choice.get("text", "")


def add_openai_compatible_args(parser):
    parser.add_argument("--api-base", help="OpenAI-compatible API base, e.g. http://localhost:8000/v1", required=True)
    parser.add_argument("--api-key", help="API key for the OpenAI-compatible server", default="EMPTY")
    parser.add_argument("--model", help="Model name to send in API requests", required=True)
    parser.add_argument(
        "--endpoint",
        help="OpenAI-compatible endpoint to use. Use completions for raw prompts or models without a chat template.",
        choices=["chat", "completions"],
        default="chat",
    )
    parser.add_argument("--system-prompt", help="Optional system prompt for chat-completions requests")
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", help="Maximum number of new tokens to generate", type=int, default=100)
    parser.add_argument("--stop", help="Stop string. Can be supplied multiple times.", action="append")
    parser.add_argument(
        "--extra-body-json",
        help='Extra JSON object merged into each request, e.g. \'{"reasoning_effort":"none"}\'.',
    )
    parser.add_argument("--request-timeout", help="HTTP request timeout in seconds", type=float, default=600.0)
    parser.add_argument("--max-retries", help="Number of HTTP retries for transient failures", type=int, default=2)
    parser.add_argument("--retry-sleep", help="Seconds to sleep between retries", type=float, default=5.0)
    parser.add_argument("--num-workers", help="Number of concurrent API requests to send", type=int, default=1)
    parser.add_argument(
        "--keep-thinking",
        help="Keep <think>...</think> blocks in model_answer. By default they are stripped for scoring.",
        action="store_true",
    )
