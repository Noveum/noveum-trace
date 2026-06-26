from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import TYPE_CHECKING, Any

from noveum_trace.guard.exceptions import NoveumGuardBlocked
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.types import ParsedRequest, ParsedResponse, PolicyContext
from noveum_trace.utils.llm_utils import estimate_cost, estimate_token_count

if TYPE_CHECKING:
    from noveum_trace.guard.decision import PolicyDecision
    from noveum_trace.guard.engine import PolicyEngine

logger = logging.getLogger(__name__)


class NoveumCrewAIInterceptor:
    """Secondary Guard integration for CrewAI — raise-only, no transport patching.

    Primary recommendation: use http_client() via crewai.LLM(client_params=...) so
    NoveumTransport handles enforcement transparently. Use this interceptor only when
    the transport approach is unavailable.

    Because CrewAI callbacks cannot return a synthetic response, block = raise
    NoveumGuardBlocked. The caller must catch it.

    Usage:
        interceptor = NoveumCrewAIInterceptor(engine, ctx)
        call_id, ran = interceptor.before_llm_call(payload)   # raises on block
        response = llm.call(payload)
        interceptor.after_llm_call(call_id, payload, response, ran)
    """

    def __init__(self, engine: PolicyEngine, context: PolicyContext) -> None:
        self._engine = engine
        self._context = context
        # Maps call_id -> PolicyContext for calls currently in flight.
        # A dict (rather than a single attribute) keeps each concurrent call's
        # context independent so post_call always reads the right ctx.
        self._pending_ctx: dict[str, PolicyContext] = {}

    def before_llm_call(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[tuple[AbstractPolicy, PolicyDecision]]]:
        """Run pre_call on all policies.

        Returns (call_id, ran) so after_llm_call can pass call_id for correct
        reconciliation. Raises NoveumGuardBlocked on block (engine has already
        rolled back any reservations).
        """
        ctx = dataclasses.replace(self._context, call_id=str(uuid.uuid4()))
        call_id = ctx.call_id
        parsed_req = self._parse_request(payload)

        block, ran = self._engine.pre_call(parsed_req, ctx)
        if block is not None:
            raise NoveumGuardBlocked(block.policy_name, block.reason, block)

        # Stash ctx keyed by call_id so after_llm_call can retrieve it
        # independently of any other concurrent call.
        self._pending_ctx[call_id] = ctx
        return call_id, ran

    def after_llm_call(
        self,
        call_id: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
    ) -> None:
        """Run post_call on all policies that fired in before_llm_call.

        `call_id` must be the value returned by before_llm_call.
        `ran` must be the list returned by before_llm_call — it carries each
        policy's pre-decision (including reserved_usd in state) so reconcile works.
        Raises NoveumGuardBlocked if any policy blocks in post.
        """
        ctx = self._pending_ctx.pop(call_id, None)
        if ctx is None:
            logger.warning(
                "after_llm_call: call_id %r not found in _pending_ctx — "
                "mismatched or already processed; skipping post_call",
                call_id,
            )
            return
        parsed_resp = self._parse_response(payload, response)

        post_block = self._engine.post_call(parsed_resp, ctx, ran)
        if post_block is not None:
            raise NoveumGuardBlocked(
                post_block.policy_name, post_block.reason, post_block
            )

    def on_llm_call_failed(
        self,
        call_id: str,
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
    ) -> None:
        """Release reservations when the LLM call raises before after_llm_call.

        Call this from an except block so in-flight budget reservations are not
        stranded when the provider raises before returning a response.
        Removes call_id from _pending_ctx to prevent memory leaks.
        """
        ctx = self._pending_ctx.pop(call_id, None)
        if ctx is None:
            return
        self._engine.release_all(ctx, ran)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_request(payload: dict[str, Any]) -> ParsedRequest:
        messages = payload.get("messages", [])
        model = payload.get("model", "")
        provider = payload.get("provider", "")
        estimated = estimate_token_count(
            messages, model=model, provider=provider or None
        )
        return ParsedRequest(
            provider=provider,
            model=payload.get("model", ""),
            messages=messages,
            stream=bool(payload.get("stream", False)),
            max_tokens=payload.get("max_tokens"),
            estimated_input_tokens=estimated,
            raw_body=b"",
        )

    @staticmethod
    def _parse_response(
        payload: dict[str, Any], response: dict[str, Any]
    ) -> ParsedResponse:

        usage = response.get("usage", {})
        model = response.get("model", payload.get("model", ""))
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        return ParsedResponse(
            model=model,
            text=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens)["total_cost"],
        )
