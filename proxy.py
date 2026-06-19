"""
enowxai-proxy: Lightweight OpenAI-compat proxy for Firecrawl → enowxai.

Transforms requests that use response_format/json_schema (Vercel AI SDK generateObject)
into plain chat completions that enowxai/kiro can handle:
  1. Strips response_format entirely
  2. Injects a system message instructing JSON-only output with the schema
  3. Forces stream: false
  4. Forwards to upstream enowxai
"""
import json
import os
import re
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://enowxai.waterflai.my.id/v1")
client = httpx.AsyncClient(timeout=120.0)


def build_schema_system_message(response_format: dict) -> str | None:
    """Extract JSON schema from response_format and build a system instruction."""
    if not response_format:
        return None
    fmt_type = response_format.get("type", "")
    if fmt_type == "json_schema":
        js = response_format.get("json_schema", {})
        schema = js.get("schema", {})
        schema_str = json.dumps(schema, separators=(",", ":"))
        return (
            "You MUST respond with ONLY valid JSON that strictly matches this schema. "
            "No explanation, no markdown, no code blocks — raw JSON only.\n"
            f"Schema: {schema_str}"
        )
    elif fmt_type == "json_object":
        return "You MUST respond with ONLY valid JSON. No explanation, no markdown, no code blocks — raw JSON only."
    return None


def transform_body(body: dict) -> dict:
    """Strip response_format, inject system message, force stream:false."""
    body = dict(body)  # shallow copy

    # Force non-streaming
    body["stream"] = False

    # Extract response_format before removing it
    response_format = body.pop("response_format", None)

    # Build system instruction from schema
    system_instruction = build_schema_system_message(response_format)

    if system_instruction:
        messages = body.get("messages", [])
        # Check if there's already a system message
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            # Prepend to existing system message
            new_messages = []
            for m in messages:
                if m.get("role") == "system":
                    new_messages.append({
                        "role": "system",
                        "content": system_instruction + "\n\n" + m.get("content", "")
                    })
                else:
                    new_messages.append(m)
            body["messages"] = new_messages
        else:
            # Inject as first message
            body["messages"] = [{"role": "system", "content": system_instruction}] + messages

    return body


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"{UPSTREAM}/{path}"
    headers = dict(request.headers)
    # Remove host header — httpx sets it automatically
    headers.pop("host", None)
    headers.pop("content-length", None)  # will be recalculated

    # Only transform POST to chat/completions
    if request.method == "POST" and "chat/completions" in path:
        try:
            body = await request.json()
            had_response_format = "response_format" in body
            body = transform_body(body)
            content = json.dumps(body).encode()
            headers["content-type"] = "application/json"
            import logging
            logging.getLogger("uvicorn").info(
                f"[proxy] transformed: had_response_format={had_response_format} "
                f"stream={body.get('stream')} model={body.get('model')} "
                f"msgs={len(body.get('messages', []))}"
            )
        except Exception as e:
            import logging
            logging.getLogger("uvicorn").error(f"[proxy] transform error: {e}")
            content = await request.body()
    else:
        content = await request.body()

    resp = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=content,
    )

    # For chat/completions POST: strip markdown code blocks from content
    if request.method == "POST" and "chat/completions" in path:
        try:
            resp_body = resp.json()
            for choice in resp_body.get("choices", []):
                msg = choice.get("message", {})
                text = msg.get("content", "")
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped.startswith("```"):
                        lines = stripped.split("\n")
                        lines = lines[1:]  # drop opening fence line
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]  # drop closing fence
                        msg["content"] = "\n".join(lines).strip()
            return JSONResponse(content=resp_body, status_code=resp.status_code)
        except Exception:
            pass  # fall through to raw response

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
