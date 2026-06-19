"""
enowxai-proxy: Lightweight 0penAI-compat proxy for Firecrawl → enowxai.

Firecrawl uses Vercel AI SDK's generateObject which sends structured output
requests using TOOL CALLING (not response_format). The tool is always named
"json" and contains the full JSON schema.

enowxai/kiro doesn't support tool calling, so:
1. We strip the tools/tool_choice from the request
2. We inject a system message with the schema and a JSON-only instruction
3. We force stream: false
4. We take the plain text response and synthesize a tool_calls response
   so the Vercel AI SDK can parse it correctly

This makes Firecrawl's extract feature work without OpenRouter.
"""
import json
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://enowxai.waterflai.my.id/v1")
client = httpx.AsyncClient(timeout=120.0)


def extract_json_schema_from_tools(tools: list) -> dict | None:
    """Extract the JSON schema from Vercel AI SDK's tool-based generateObject request."""
    for tool in tools or []:
        fn = tool.get("function", {})
        if fn.get("name") == "json":
            return fn.get("parameters")
    return None


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        return "\n".join(lines).strip()
    return stripped


def fix_nulls(obj):
    """Replace null values with empty strings for schema compatibility."""
    if isinstance(obj, dict):
        return {k: ("" if v is None else fix_nulls(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_nulls(i) for i in obj]
    return obj


def transform_request(body: dict) -> tuple[dict, bool]:
    """
    Transform a tool-calling generateObject request into a plain chat request.
    Returns (transformed_body, had_tool_schema).
    """
    body = dict(body)
    body["stream"] = False  # force non-streaming

    # Normalize any 0penAI model names to kiro/claude-haiku-4.5
    # (Firecrawl hardcodes gpt-4o-mini/gpt-4.1-mini but enowxai doesn't have them)
    model = body.get("model", "")
    if any(m in model for m in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4"]):
        body["model"] = "kiro/claude-haiku-4.5"

    # Extract tool schema if present
    tools = body.pop("tools", None)
    body.pop("tool_choice", None)

    schema = extract_json_schema_from_tools(tools)
    had_tool_schema = schema is not None

    if schema:
        schema_str = json.dumps(schema, separators=(",", ":"))
        system_instruction = (
            "You MUST respond with ONLY valid JSON that strictly matches this schema. "
            "No explanation, no markdown code blocks, no extra text — raw JSON only.\n"
            f"Schema: {schema_str}"
        )
        messages = body.get("messages", [])
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
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
            body["messages"] = [{"role": "system", "content": system_instruction}] + messages

    return body, had_tool_schema


def synthesize_tool_call_response(resp_body: dict, tool_name: str = "json") -> dict:
    """
    Take the upstream response (plain text JSON in message.content) and
    synthesize it into a tool_calls response that the Vercel AI SDK expects.
    """
    for choice in resp_body.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            clean = strip_markdown_fences(content)
            try:
                parsed = json.loads(clean)
                fixed = fix_nulls(parsed)
                args_str = json.dumps(fixed, separators=(",", ":"))
            except Exception:
                args_str = clean

            # Replace plain content with tool_calls
            msg["content"] = None
            msg["tool_calls"] = [{
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_str
                }
            }]
            choice["finish_reason"] = "tool_calls"
    return resp_body


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"{UPSTREAM}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    had_tool_schema = False

    if request.method == "POST" and "chat/completions" in path:
        try:
            body = await request.json()
            body, had_tool_schema = transform_request(body)
            content = json.dumps(body).encode()
            headers["content-type"] = "application/json"
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

    # For chat/completions POST with tool schema: synthesize tool_calls response
    if request.method == "POST" and "chat/completions" in path:
        try:
            resp_body = resp.json()
            if had_tool_schema:
                resp_body = synthesize_tool_call_response(resp_body)
            else:
                # No tool schema — just strip markdown fences from plain responses
                for choice in resp_body.get("choices", []):
                    msg = choice.get("message", {})
                    text = msg.get("content", "")
                    if isinstance(text, str) and text.strip().startswith("```"):
                        msg["content"] = strip_markdown_fences(text)
            return JSONResponse(content=resp_body, status_code=resp.status_code)
        except Exception:
            pass

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
