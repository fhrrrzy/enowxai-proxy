"""
enowxai-proxy: Lightweight OpenAI-compat proxy for Firecrawl → enowxai.

Firecrawl uses Vercel AI SDK's generateObject which sends structured output
requests. With newer Vercel AI SDK versions it uses the /v1/responses
(Responses API) endpoint. Older versions use /v1/chat/completions with
tool calling.

enowxai/kiro doesn't support tool calling or the Responses API, so:
1. We intercept /v1/responses and /v1/chat/completions
2. We strip tools/tool_choice, inject schema as system message
3. We force stream: false
4. We convert the plain response → tool_calls or responses API format
   so the Vercel AI SDK can parse it correctly
"""
import json
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://enowxai.waterflai.my.id/v1")
client = httpx.AsyncClient(timeout=120.0)

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM}

def extract_json_schema_from_tools(tools: list) -> dict | None:
    for tool in tools or []:
        fn = tool.get("function", {})
        if fn.get("name") == "json":
            return fn.get("parameters")
    return None

def extract_json_schema_from_responses_tools(tools: list) -> dict | None:
    """Extract schema from Responses API tool format."""
    for tool in tools or []:
        if tool.get("type") == "function":
            fn = tool.get("function", {}) if "function" in tool else tool
            name = fn.get("name") or tool.get("name", "")
            if name == "json":
                return fn.get("parameters") or tool.get("parameters")
        # Also handle direct function objects
        if tool.get("name") == "json":
            return tool.get("parameters")
    return None

def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped

def fix_nulls(obj):
    if isinstance(obj, dict):
        return {k: ("" if v is None else fix_nulls(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_nulls(i) for i in obj]
    return obj

def inject_schema_into_messages(messages: list, schema: dict) -> list:
    schema_str = json.dumps(schema, separators=(",", ":"))
    system_instruction = (
        "You MUST respond with ONLY valid JSON that strictly matches this schema. "
        "No explanation, no markdown code blocks, no extra text — raw JSON only.\n"
        f"Schema: {schema_str}"
    )
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
        return new_messages
    else:
        return [{"role": "system", "content": system_instruction}] + messages

def normalize_model(model: str) -> str:
    if any(m in model for m in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4"]):
        return "kiro/claude-haiku-4.5"
    return model

def transform_chat_completions(body: dict) -> tuple[dict, bool]:
    body = dict(body)
    body["stream"] = False
    body["model"] = normalize_model(body.get("model", ""))

    tools = body.pop("tools", None)
    body.pop("tool_choice", None)
    schema = extract_json_schema_from_tools(tools)
    had_tool_schema = schema is not None

    if schema:
        body["messages"] = inject_schema_into_messages(body.get("messages", []), schema)

    return body, had_tool_schema

def synthesize_tool_call_response(resp_body: dict, tool_name: str = "json") -> dict:
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

def responses_to_chat_completions(body: dict) -> tuple[dict, bool]:
    """
    Convert a Responses API request to chat/completions format.
    Responses API uses 'input' (array or string) instead of 'messages'.
    """
    body = dict(body)
    body["stream"] = False
    body["model"] = normalize_model(body.get("model", ""))

    # Extract schema from tools
    tools = body.pop("tools", None)
    body.pop("tool_choice", None)
    schema = extract_json_schema_from_responses_tools(tools)
    had_tool_schema = schema is not None

    # Convert 'input' → 'messages'
    input_val = body.pop("input", None)
    if input_val is None:
        messages = body.pop("messages", [])
    elif isinstance(input_val, str):
        messages = [{"role": "user", "content": input_val}]
    elif isinstance(input_val, list):
        # Responses API uses {role, content} same as chat
        messages = input_val
    else:
        messages = []

    # Remove Responses API-specific fields
    for key in ["previous_response_id", "store", "metadata", "truncation", "reasoning",
                "include", "temperature", "top_p", "max_output_tokens", "text", "modalities"]:
        body.pop(key, None)

    # Map max_output_tokens → max_tokens
    if "max_output_tokens" in body:
        body["max_tokens"] = body.pop("max_output_tokens")

    if schema:
        messages = inject_schema_into_messages(messages, schema)

    body["messages"] = messages
    return body, had_tool_schema, tools

def synthesize_responses_api_response(plain_resp: dict, had_tool_schema: bool, original_tools) -> dict:
    """
    Convert a chat/completions response back to Responses API format.
    """
    choices = plain_resp.get("choices", [])
    content_text = ""
    tool_calls_out = []

    for choice in choices:
        msg = choice.get("message", {})
        raw_content = msg.get("content", "") or ""

        if had_tool_schema and isinstance(raw_content, str) and raw_content.strip():
            clean = strip_markdown_fences(raw_content)
            try:
                parsed = json.loads(clean)
                fixed = fix_nulls(parsed)
                args_str = json.dumps(fixed, separators=(",", ":"))
            except Exception:
                args_str = clean

            tool_calls_out.append({
                "type": "function_call",
                "id": "call_json",
                "call_id": "call_json",
                "name": "json",
                "arguments": args_str
            })
        else:
            content_text = strip_markdown_fences(raw_content) if isinstance(raw_content, str) else raw_content

    # Build Responses API output
    output = []
    if tool_calls_out:
        output.extend(tool_calls_out)
    elif content_text:
        output.append({
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content_text}]
        })

    return {
        "id": plain_resp.get("id", "resp_001"),
        "object": "response",
        "created_at": plain_resp.get("created", 0),
        "model": plain_resp.get("model", ""),
        "status": "completed",
        "output": output,
        "usage": plain_resp.get("usage", {}),
    }

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    import logging
    logger = logging.getLogger("uvicorn")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    had_tool_schema = False
    is_responses_api = False
    original_tools = None

    if request.method == "POST" and path == "responses":
        # Responses API → convert to chat/completions
        is_responses_api = True
        try:
            body = await request.json()
            logger.info(f"[proxy] /v1/responses request, model={body.get('model')}, tools={[t.get('name') or t.get('function',{}).get('name') for t in (body.get('tools') or [])]}")
            body, had_tool_schema, original_tools = responses_to_chat_completions(body)
            # Route to chat/completions upstream
            upstream_url = f"{UPSTREAM}/chat/completions"
            content = json.dumps(body).encode()
            headers["content-type"] = "application/json"

            resp = await client.request(
                method="POST",
                url=upstream_url,
                headers=headers,
                content=content,
            )
            logger.info(f"[proxy] upstream status={resp.status_code}")
            resp_body = resp.json()
            result = synthesize_responses_api_response(resp_body, had_tool_schema, original_tools)
            r = JSONResponse(content=result, status_code=resp.status_code)
            r.headers["X-Proxy-Hit"] = "1"
            return r
        except Exception as e:
            logger.error(f"[proxy] responses error: {e}", exc_info=True)
            return JSONResponse(content={"error": str(e)}, status_code=500)

    elif request.method == "POST" and "chat/completions" in path:
        try:
            body = await request.json()
            body, had_tool_schema = transform_chat_completions(body)
            content = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        except Exception as e:
            logger.error(f"[proxy] transform error: {e}")
            content = await request.body()

        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            headers=headers,
            content=content,
        )

        try:
            resp_body = resp.json()
            if had_tool_schema:
                resp_body = synthesize_tool_call_response(resp_body)
            else:
                for choice in resp_body.get("choices", []):
                    msg = choice.get("message", {})
                    text = msg.get("content", "")
                    if isinstance(text, str) and text.strip().startswith("```"):
                        msg["content"] = strip_markdown_fences(text)
            r = JSONResponse(content=resp_body, status_code=resp.status_code)
            r.headers["X-Proxy-Hit"] = "1"
            return r
        except Exception:
            pass

    else:
        content = await request.body()
        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            headers=headers,
            content=content,
        )

    r = Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
    r.headers["X-Proxy-Hit"] = "1"
    return r
