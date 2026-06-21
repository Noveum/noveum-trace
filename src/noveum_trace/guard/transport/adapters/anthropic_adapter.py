from __future__ import annotations

import json
from typing import ClassVar

import httpx

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.transport.adapters.base import ProviderAdapter
from noveum_trace.guard.types import BlockResponseMode, ParsedRequest, ParsedResponse
from noveum_trace.utils.llm_utils import estimate_cost, estimate_token_count


class AnthropicAdapter(ProviderAdapter):
    provider_name: ClassVar[str] = "anthropic"
    host_patterns: ClassVar[list[str]] = ["api.anthropic.com"]
    body_signature_keys: ClassVar[list[str]] = ["model", "messages", "max_tokens"]
    # The Anthropic SDK always sends this header; required for body-fallback
    # matching so OpenAI bodies carrying max_tokens aren't misrouted here.
    header_signatures: ClassVar[list[str]] = ["anthropic-version"]

    def parse_request(self, req: httpx.Request) -> ParsedRequest:
        body = json.loads(req.content)
        messages = body.get("messages", [])
        if "system" in body:
            messages = [{"role": "system", "content": body["system"]}] + messages
        model = body.get("model", "")
        estimated = estimate_token_count(messages, model=model, provider="anthropic")
        return ParsedRequest(
            provider="anthropic",
            model=body.get("model", ""),
            messages=messages,
            stream=bool(body.get("stream", False)),
            max_tokens=body.get("max_tokens"),
            estimated_input_tokens=estimated,
            raw_body=req.content,
        )

    def parse_response(
        self, req: httpx.Request, resp: httpx.Response
    ) -> ParsedResponse:
        body = json.loads(resp.content)
        usage = body.get("usage", {})
        model = body.get("model", "")
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        contents = body.get("content", [])
        text = next((c.get("text") for c in contents if c.get("type") == "text"), None)
        return ParsedResponse(
            model=model,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens)["total_cost"],
        )

    def synthetic_block_response(
        self,
        req: httpx.Request,
        decision: PolicyDecision,
        mode: BlockResponseMode = BlockResponseMode.provider_error,
    ) -> httpx.Response:
        body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": decision.reason or "Request blocked by Noveum Guard",
                },
            }
        ).encode()
        return httpx.Response(
            status_code=403, content=body, headers={"content-type": "application/json"}
        )
