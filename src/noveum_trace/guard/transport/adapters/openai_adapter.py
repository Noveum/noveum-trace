from __future__ import annotations

import json
from typing import ClassVar

import httpx

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.transport.adapters.base import ProviderAdapter
from noveum_trace.guard.types import BlockResponseMode, ParsedRequest, ParsedResponse
from noveum_trace.utils.llm_utils import estimate_cost, estimate_token_count


class OpenAIAdapter(ProviderAdapter):
    provider_name: ClassVar[str] = "openai"
    host_patterns: ClassVar[list[str]] = [
        "api.openai.com",
        "api.groq.com",
        "api.together.xyz",
        "api.perplexity.ai",
        "api.fireworks.ai",
        "api.cerebras.ai",
        "api.deepseek.com",
    ]
    body_signature_keys: ClassVar[list[str]] = ["model", "messages"]

    def parse_request(self, req: httpx.Request) -> ParsedRequest:
        body = json.loads(req.content)
        messages = body.get("messages", [])
        model = body.get("model", "")
        estimated = estimate_token_count(messages, model=model, provider="openai")
        return ParsedRequest(
            provider="openai",
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
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        choices = body.get("choices", [])
        text = choices[0].get("message", {}).get("content") if choices else None
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
        # 403 is outside the Stainless SDK retry set (408/409/429/≥500) so
        # OpenAI SDK raises PermissionDeniedError without silently retrying.
        body = json.dumps(
            {
                "error": {
                    "message": decision.reason or "Request blocked by Noveum Guard",
                    "type": "policy_blocked",
                    "code": "policy_blocked",
                }
            }
        ).encode()
        return httpx.Response(
            status_code=403, content=body, headers={"content-type": "application/json"}
        )
