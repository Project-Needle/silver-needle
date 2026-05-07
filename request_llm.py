import requests
import json
import time

def safe_post(resp):
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            raise Exception(f"HTTP {resp.status_code}: {resp.text}")

        raise Exception(
            f"LLM API Error:\n"
            f"HTTP {resp.status_code}\n"
            f"message: {err.get('error', {}).get('message')}\n"
            f"type: {err.get('error', {}).get('type')}\n"
            f"raw: {err}"
        )

    return resp


# =============================================================================
# OpenAI-compatible  /v1/chat/completions
# =============================================================================

def chat_completions_non_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int =120,
    logprob: bool = False
):
    """
    Simple OpenAI-compatible chat/completions client.

    Args:
        base_url: e.g. https://api.openai.com
        api_key: Bearer token
        model: model name
        messages: [{"role": "user", "content": "..."}]
        max_tokens: output limit
        stream: whether to stream
    """

    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    # 非流式的时候openai-compatible chat/completions是支持logprob的
    if logprob:
        payload["logprobs"] = True
        payload["top_logprobs"] = 2

    # =========================
    # 非流式
    # =========================
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp = safe_post(resp)
    return resp.json()

def chat_completions_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    timeout: int = 120,
    temperature: float = 0.7,
):
    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    
    # =========================
    # 流式（SSE）
    # =========================
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout
    )

    if not resp.ok:
        try:
            print(resp.json())
        except:
            print(resp.text)
        raise Exception("Streaming request failed")

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        # SSE 格式：data: {...}
        if line.startswith("data: "):
            data = line[len("data: "):]

            # 结束标志
            if data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            yield chunk


# =============================================================================
# Anthropic  /v1/messages
# =============================================================================

def anthropic_messages_non_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = 120,
    system: str = None,
):
    """
    Anthropic Messages API (non-streaming).

    Args:
        api_key:   Anthropic API key (x-api-key header)
        model:     e.g. "claude-opus-4-7", "claude-sonnet-4-6"
        messages:  [{"role": "user"|"assistant", "content": "..."}]
        max_tokens: output limit
        temperature: sampling temperature
        system:    optional system prompt string

    Returns raw JSON response dict.
    Response shape:
      {
        "id": "msg_...",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "..."}],
        "model": "...",
        "stop_reason": "end_turn" | "max_tokens" | ...,
        "usage": {"input_tokens": N, "output_tokens": N}
      }
    """
    url = f"{base_url}/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        payload["system"] = system

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    safe_post(resp)
    return resp.json()


def anthropic_messages_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = 120,
    system: str = None,
):
    """
    Anthropic Messages API (streaming).

    Yields parsed SSE event dicts. Relevant event types:
      - "content_block_delta":  {"delta": {"type": "text_delta", "text": "..."}}
      - "message_delta":        {"delta": {"stop_reason": "end_turn", ...}, "usage": {...}}
      - "message_start":        {"message": {"usage": {"input_tokens": N}, ...}}
      - "message_stop":         (signals end)

    Anthropic SSE format differs from OpenAI:
      event: content_block_delta
      data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}

    Note: some third-party providers that implement the Anthropic-compatible API buffer the
    entire response server-side before sending, despite returning correct SSE headers
    (Transfer-Encoding: chunked, Content-Type: text/event-stream). In that case all events
    arrive in a single network chunk and there is no incremental token delivery — this is a
    provider-side limitation and cannot be worked around on the client.
    """
    url = f"{base_url}/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    )

    if not resp.ok:
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
        raise Exception(f"Anthropic streaming request failed: HTTP {resp.status_code}")

    current_event_type = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            current_event_type = None
            continue

        if line.startswith("event: "):
            current_event_type = line[len("event: "):].strip()
            continue

        if line.startswith("data: "):
            data = line[len("data: "):].strip()
            if data == "[DONE]" or current_event_type == "message_stop":
                break
            try:
                chunk = json.loads(data)
                chunk["_event"] = current_event_type
                yield chunk
            except json.JSONDecodeError:
                continue


# one-time unified test function
def request_llm(
    base_url: str,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    stream: bool = False,
    temperature: float = 0.7,
    logprob: bool = False,
    api_style: str = "openai"
):
    messages = [{"role": "user", "content": message} for message in messages]
    if api_style.strip().lower() == "openai":
        # openai-compatible
        if stream:
            for chunk in chat_completions_stream(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                print(chunk, end="\n\n", flush=True)
        else:
            result = chat_completions_non_stream(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprob=logprob
            )
            print(result)
    else:
        # anthropic-compatible
        if stream:
            for chunk in anthropic_messages_stream(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                print(chunk, end="\n\n", flush=True)
        else:
            result = anthropic_messages_non_stream(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(result)


# =============================================================================
# Unified non-streaming call for probe scripts
# =============================================================================

def call_api(messages: list, client_cfg: dict, temperature: float, timeout: int = 120) -> str:
    """
    Single non-streaming call that works for both API styles.

    client_cfg keys:
      base_url   : API base URL
      api_key    : API key
      model      : model name
      max_tokens : (optional) output limit, default 1024
      api_style  : "openai" (default) | "anthropic"
      system     : (optional, anthropic only) system prompt string

    Returns the assistant text content as a plain string.
    """
    api_style = client_cfg.get("api_style", "openai").lower()
    max_tokens = client_cfg.get("max_tokens", 1024)

    if api_style == "anthropic":
        raw = anthropic_messages_non_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            system=client_cfg.get("system"),
        )
        # Anthropic response: content is a list of blocks
        try:
            return "".join(
                block["text"] for block in raw["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError):
            return ""
    else:
        raw = chat_completions_non_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        try:
            return raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return ""



# =============================================================================
# Raw response call for structure analysis
# =============================================================================

def call_api_raw(messages: list, client_cfg: dict, temperature: float, timeout: int = 120) -> dict:
    """
    Like call_api(), but returns the full raw response dict instead of extracted text.

    client_cfg keys are identical to call_api().

    Return value shape:
      {
        "raw":     <full JSON response from provider>,
        "content": <extracted text content, same as call_api()>,
        "latency": <float, seconds>
      }
    """
    api_style = client_cfg.get("api_style", "openai").lower()
    max_tokens = client_cfg.get("max_tokens", 1024)

    t0 = time.time()
    if api_style == "anthropic":
        raw = anthropic_messages_non_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            system=client_cfg.get("system"),
        )
        latency = time.time() - t0
        try:
            content = "".join(
                block["text"] for block in raw["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError):
            content = ""
    else:
        raw = chat_completions_non_stream(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        latency = time.time() - t0
        try:
            content = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            content = ""

    return {"raw": raw, "content": content, "latency": round(latency, 2)}


# in order to determine whether the server truly supports real-time streaming output or 
# buffering streaming.
def raw_stream_test(base_url, api_key, model):
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        "max_tokens": 100,
        "temperature": 0,
        "stream": True,
    }

    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
    print(f"HTTP {resp.status_code}, headers: {dict(resp.headers)}\n")

    # iter_content with chunk_size=None: yield whatever arrives from socket immediately
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        print(f"[+{time.time()-t0:.2f}s] received {len(chunk)} bytes: {chunk[:80]!r}")

if __name__ == "__main__":
    base_url = "https://localhost:3001"
    api_key = "sk-xxx"
    model = "claude-sonnet-4-6"
    messages = ["你好"]
    request_llm(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=128,
        stream=False,
        api_style="anthropic"
    )