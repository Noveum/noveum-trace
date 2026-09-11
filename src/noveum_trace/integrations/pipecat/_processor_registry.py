"""Session-local Pipecat processor identity and role registry.

Processor display names are not unique. The registry therefore treats the actual
processor object as identity and uses exact names only when a metric record does
not carry an object reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

PROCESSOR_ROLE_LLM = "llm"
PROCESSOR_ROLE_STT = "stt"
PROCESSOR_ROLE_TTS = "tts"
PROCESSOR_ROLE_TURN = "turn"
PROCESSOR_ROLE_OTHER = "other"

_KNOWN_ROLES = {
    PROCESSOR_ROLE_LLM,
    PROCESSOR_ROLE_STT,
    PROCESSOR_ROLE_TTS,
    PROCESSOR_ROLE_TURN,
    PROCESSOR_ROLE_OTHER,
}


@dataclass(frozen=True)
class ProcessorRecord:
    """One processor registration. The object reference is never exported."""

    processor: Any = field(compare=False, repr=False)
    key: str
    name: str
    class_name: str
    roles: frozenset[str]

    @property
    def role(self) -> str:
        """Compatibility label for diagnostics expecting one role."""
        return next(iter(self.roles)) if len(self.roles) == 1 else "multiple"

    def has_role(self, role: str) -> bool:
        return role in self.roles


class ProcessorRegistry:
    """Resolve Pipecat processors by exact object identity and explicit role."""

    def __init__(self) -> None:
        self._records: list[ProcessorRecord] = []
        self._sequences: dict[str, int] = {}
        self._explicit_roles: dict[int, set[str]] = {}

    @staticmethod
    def _display_name(processor: Any) -> str:
        name = getattr(processor, "name", None)
        if isinstance(name, str) and name:
            return name
        return type(processor).__name__

    @staticmethod
    def infer_role(processor: Any) -> str:
        """Classify from service base classes, never display-name substrings."""
        roles = ProcessorRegistry.infer_roles(processor)
        return next(iter(roles)) if len(roles) == 1 else PROCESSOR_ROLE_OTHER

    @staticmethod
    def infer_roles(processor: Any) -> set[str]:
        """Return every service capability exposed by the processor's base types."""
        if processor is None:
            return {PROCESSOR_ROLE_OTHER}
        mro_names = {base.__name__ for base in type(processor).__mro__}
        roles: set[str] = set()
        if "LLMService" in mro_names:
            roles.add(PROCESSOR_ROLE_LLM)
        if "STTService" in mro_names:
            roles.add(PROCESSOR_ROLE_STT)
        if "TTSService" in mro_names:
            roles.add(PROCESSOR_ROLE_TTS)
        if mro_names.intersection(
            {"TurnAnalyzer", "BaseTurnAnalyzer", "UserTurnProcessor"}
        ):
            roles.add(PROCESSOR_ROLE_TURN)
        return roles or {PROCESSOR_ROLE_OTHER}

    def set_explicit_role(self, processor: Any, role: str) -> ProcessorRecord:
        """Register an escape-hatch role for a custom processor."""
        if role not in _KNOWN_ROLES - {PROCESSOR_ROLE_OTHER}:
            raise ValueError(f"Unsupported Pipecat processor role: {role}")
        self._explicit_roles.setdefault(id(processor), set()).add(role)
        existing = self.get(processor)
        if existing is not None:
            if existing.has_role(role):
                return existing
            self._records.remove(existing)
            return self.register(processor, roles=set(existing.roles) | {role})
        return self.register(processor, roles={role})

    def register(
        self,
        processor: Any,
        role: Optional[str] = None,
        roles: Optional[set[str]] = None,
    ) -> ProcessorRecord:
        """Return the existing record or add a session-local processor record."""
        existing = self.get(processor)
        if existing is not None:
            requested_roles = set(roles or ())
            if role is not None:
                requested_roles.add(role)
            if not requested_roles or requested_roles.issubset(existing.roles):
                return existing
            self._records.remove(existing)
            roles = set(existing.roles) | requested_roles

        resolved_roles = set(roles or ())
        if role is not None:
            resolved_roles.add(role)
        resolved_roles.update(self._explicit_roles.get(id(processor), set()))
        if not resolved_roles:
            resolved_roles = self.infer_roles(processor)
        name = self._display_name(processor)
        role_label = "+".join(sorted(resolved_roles))
        sequence_key = f"{role_label}:{name}"
        sequence = self._sequences.get(sequence_key, 0) + 1
        self._sequences[sequence_key] = sequence
        record = ProcessorRecord(
            processor=processor,
            key=f"processor-{role_label}-{sequence}",
            name=name,
            class_name=type(processor).__name__,
            roles=frozenset(resolved_roles),
        )
        self._records.append(record)
        return record

    def get(self, processor: Any) -> Optional[ProcessorRecord]:
        if processor is None:
            return None
        for record in self._records:
            if record.processor is processor:
                return record
        return None

    def records_for_name(
        self, name: Optional[str], role: Optional[str] = None
    ) -> list[ProcessorRecord]:
        """Return every exact-name match; callers must handle ambiguity."""
        if not name:
            return []
        return [
            record
            for record in self._records
            if record.name == name and (role is None or record.has_role(role))
        ]

    def records_for_role(self, role: str) -> list[ProcessorRecord]:
        return [record for record in self._records if record.has_role(role)]

    def resolve_metric_processors(
        self,
        source: Any,
        reported_name: Optional[str],
        required_role: Optional[str],
    ) -> list[ProcessorRecord]:
        """Resolve a metric source without guessing from partial names."""
        if source is not None:
            source_record = self.register(source)
            if required_role is not None and not source_record.has_role(required_role):
                if source_record.roles != {PROCESSOR_ROLE_OTHER}:
                    return []
            elif source_record.roles != {PROCESSOR_ROLE_OTHER}:
                return [source_record]

        named = self.records_for_name(reported_name, required_role)
        if named:
            return named

        if required_role is not None:
            return self.records_for_role(required_role)
        return []

    def clear(self) -> None:
        self._records.clear()
        self._sequences.clear()
        self._explicit_roles.clear()
