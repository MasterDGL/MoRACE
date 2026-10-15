"""Immutable scientific settings scoped to one generation call.

CLI entrypoints explicitly capture environment settings. Python generation APIs
use fixed defaults unless a runtime is supplied; process environment is never
modified when activating a runtime or a nested projection repair.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import math
import os
from types import MappingProxyType
from typing import Mapping


_NUMERIC_DEFAULTS = {
    "LATE_SELECTION_KEEP_TOPK": 1,
    "LATE_SELECTION_MAX_SCORE_DELTA": 0.0,
    "LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT": 1,
    "LATE_SELECTION_REQUIRE_NOVELTY": True,
    "LITERATURE_RECALL_RETRY_ENABLE": True,
    "LITERATURE_RECALL_BONUS": 0.30,
    "LITERATURE_RECALL_FAMILY_BONUS": 0.20,
    "LITERATURE_RECALL_MAX_SCORE_DELTA": 0.30,
}
# These two scoring constants were never environment switches in the method.
_FIXED_KEYS = frozenset({"LITERATURE_RECALL_FAMILY_BONUS", "LITERATURE_RECALL_MAX_SCORE_DELTA"})
_PREFIXES = ("STAGE2_", "CHAT_NATIVE_CORE_", "LATE_SELECTION_", "LITERATURE_RECALL_",
             "MORACE_MODEL_PATH", "LLM4SD_MODEL_PATH")
_KEYS = frozenset({
    "USE_CROSS_MODEL_PROJECTOR", "MORACE_TRUST_REMOTE_CODE", "LLM4SD_TRUST_REMOTE_CODE",
    "MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_LLAMA_ALLOW_AUTO_DEVICE_MAP",
    "MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_BAICHUAN_ALLOW_AUTO_DEVICE_MAP",
    "MORACE_GALACTICA_PLAIN_PROMPT", "LLM4SD_GALACTICA_PLAIN_PROMPT",
    "_MORACE_STAGE2_PROTOCOL_PROFILE", "_LLM4SD_STAGE2_PROTOCOL_PROFILE",
})
_ACTIVE: ContextVar[GenerationRuntime | None] = ContextVar("morace_generation_runtime", default=None)


@dataclass(frozen=True)
class GenerationRuntime:
    """A copied, read-only set of generation policy overrides."""

    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        values = {str(key): str(value) for key, value in self.environment.items()
                  if key not in _FIXED_KEYS and (key in _KEYS or str(key).startswith(_PREFIXES))}
        for key, default in _NUMERIC_DEFAULTS.items():
            if key not in values:
                continue
            try:
                value = int(values[key]) if isinstance(default, (bool, int)) else float(values[key])
                valid = math.isfinite(value) and value >= 0
                if isinstance(default, bool):
                    valid = value in {0, 1}
                if key in {"LATE_SELECTION_KEEP_TOPK", "LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT"}:
                    valid = valid and value >= 1
                if not valid:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid generation setting {key}={values[key]!r}") from exc
        object.__setattr__(self, "environment", MappingProxyType(values))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> GenerationRuntime:
        """Capture settings once at a CLI boundary, or from an explicit mapping."""
        return cls(dict(os.environ if environment is None else environment))

    def with_overrides(self, **overrides: str) -> GenerationRuntime:
        return type(self)({**self.environment, **overrides})

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)


def current_generation_runtime() -> GenerationRuntime | None:
    return _ACTIVE.get()


def science_env_get(key: str, default=None):
    """Resolve within a call; unscoped legacy helpers retain their CLI behavior."""
    runtime = _ACTIVE.get()
    return (runtime.environment if runtime is not None else os.environ).get(key, default)


def generation_option(key: str, legacy_default):
    """Use literal defaults inside runtime APIs, retaining legacy helper hooks."""
    runtime = _ACTIVE.get()
    if runtime is None:
        return legacy_default
    default = _NUMERIC_DEFAULTS[key]
    value = runtime.environment.get(key, default)
    if isinstance(default, bool):
        return bool(int(value))
    return type(default)(value)


def with_generation_runtime(function):
    """Bind one runtime to an entire public call, including nested helpers."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        runtime = kwargs.get("runtime")
        if runtime is None:
            runtime = _ACTIVE.get() or GenerationRuntime()
        if not isinstance(runtime, GenerationRuntime):
            raise TypeError("runtime must be a GenerationRuntime")
        with runtime.activate():
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def generation_overrides(**overrides: str):
    """Scope an internal repair profile without touching os.environ."""
    runtime = _ACTIVE.get() or GenerationRuntime.from_environment()
    with runtime.with_overrides(**overrides).activate() as nested:
        yield nested
