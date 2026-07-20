#!/usr/bin/env python3
"""Authorize, build, freeze, and verify private dynamic-landmark SSL inputs."""
from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import dis
import enum
import hashlib
import inspect
import io
import json
import marshal
import operator
import os
import re
import stat
import sys
import tempfile
import threading
import unicodedata
import zipfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from types import CodeType, ModuleType
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from scripts import prepare_dynamic_landmark_ssl_inputs as _canonical_entry

    raise SystemExit(_canonical_entry.main())

from scripts.build_mayo_ssl_cache import (  # noqa: E402
    authorize_committed_mayo_ssl_generation,
)
from scripts.prepare_ravdess_semantic23 import (  # noqa: E402
    audit_ravdess_inventory,
    authorize_committed_ravdess_semantic23,
)
from src.pretraining import dynamic_landmark_ssl as _dynamic_landmark_ssl  # noqa: E402,F401
from src.pretraining.dynamic_landmark_ssl_bridge import (  # noqa: E402
    build_bridge_bundles,
    freeze_bridge_stage,
    initialize_owner_only_key,
    verify_bridge_generation,
    verify_frozen_bridge_stage,
)


def _capture_trainer_authorization_marker(original: object):
    def require_original(value: object) -> bool:
        trainer = sys.modules.get("src.pretraining.dynamic_landmark_ssl")
        if (
            not isinstance(trainer, ModuleType)
            or vars(trainer).get("_AUTHORIZATION_MARKER") is not original
        ):
            raise ValueError(
                "trainer authorization marker changed after producer import"
            )
        return value is original

    return require_original


_is_original_trainer_authorization_marker = (
    _capture_trainer_authorization_marker(
        _dynamic_landmark_ssl._AUTHORIZATION_MARKER
    )
)
del _capture_trainer_authorization_marker


PRETRAINING_ROOT = ROOT / "outputs" / "dynamic_landmark" / "pretraining"
CANONICAL_MAYO_KEY = PRETRAINING_ROOT / ".mayo_ssl_hmac.key"
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PRODUCER_FILES = (
    ROOT / "src" / "pretraining" / "dynamic_landmark_ssl.py",
    ROOT / "src" / "pretraining" / "dynamic_landmark_ssl_bridge.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "prepare_ravdess_semantic23.py",
    ROOT / "scripts" / "build_mayo_ssl_cache.py",
    ROOT / "src" / "preprocessing" / "semantic_landmarks.py",
    ROOT / "src" / "preprocessing" / "openface68_semantic.py",
    ROOT / "src" / "datasets" / "dynamic_landmark.py",
)
_PRODUCER_MODULE_NAMES = (
    "src.pretraining.dynamic_landmark_ssl",
    "src.pretraining.dynamic_landmark_ssl_bridge",
    __name__,
    "scripts.prepare_ravdess_semantic23",
    "scripts.build_mayo_ssl_cache",
    "src.preprocessing.semantic_landmarks",
    "src.preprocessing.openface68_semantic",
    "src.datasets.dynamic_landmark",
)
_MAX_PRODUCER_FILE_BYTES = 4 * 1024 * 1024
_MAX_PRODUCER_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_LIVE_PRODUCER_MODULES = 16
_MAX_LIVE_PRODUCER_COMPONENTS = 2048
_MAX_LIVE_SEMANTIC_NODES = 65536
_MAX_LIVE_SEMANTIC_DEPTH = 64
_MAX_LIVE_SEMANTIC_CONTAINER = 8192
_MAX_LIVE_SEMANTIC_LEAF_BYTES = 4 * 1024 * 1024
_MAX_LIVE_SEMANTIC_TOTAL_BYTES = 32 * 1024 * 1024
_LIVE_CALL_RESULT_SEGMENT = "\0dynamic-landmark-call-result\0"
_LIVE_EXTERNAL_BEHAVIOR_SEGMENT = "\0dynamic-landmark-external-behavior\0"
_EMPTY_CODE_TEMPLATE = (lambda: None).__code__
_MAX_MAYO_CLI_CAPTURE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _AuthorizedView:
    authorization: object = dataclass_field(repr=False)
    key_file_identity_sha256: str

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.authorization, name)

    def __repr__(self) -> str:
        return "_AuthorizedView(authorization=<redacted>, key_identity=<sha256>)"


@dataclass(frozen=True)
class _AuthorizationPrivacySnapshot:
    private_key: bytes = dataclass_field(repr=False)
    cache_sha256s: tuple[str, ...] = dataclass_field(repr=False)
    sensitive_values: tuple[str, ...] = dataclass_field(repr=False)
    privacy_inventory: object | None = dataclass_field(repr=False)


def _authorization_privacy_snapshot(
    authorization: object,
    collection_name: str,
) -> _AuthorizationPrivacySnapshot:
    key = getattr(authorization, "private_key", None)
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("live authorization key is malformed")
    cache_sha256s: list[str] = []
    sensitive_values: list[str] = []
    for item in tuple(getattr(authorization, collection_name, ())):
        cache_sha256 = getattr(item, "cache_sha256", None)
        if type(cache_sha256) is str:
            cache_sha256s.append(cache_sha256)
        for field in (
            "source_sha256", "raw_filename", "filename", "session_name",
            "patient_id", "patient_identity", "subject_id",
        ):
            value = getattr(item, field, None)
            if type(value) is str:
                sensitive_values.append(value)
    return _AuthorizationPrivacySnapshot(
        private_key=key,
        cache_sha256s=tuple(cache_sha256s),
        sensitive_values=tuple(sensitive_values),
        privacy_inventory=getattr(authorization, "privacy_inventory", None),
    )


def _resolve_existing_private_directory(value: Path) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("private path parent is unavailable or unsafe") from exc
    if resolved == lexical:
        return resolved
    aliases = (
        (Path("/var"), Path("/private/var")),
        (Path("/tmp"), Path("/private/tmp")),
    ) if sys.platform == "darwin" else ()
    for source, destination in aliases:
        try:
            relative = lexical.relative_to(source)
        except ValueError:
            continue
        if resolved == destination / relative:
            return resolved
    raise ValueError("private path contains an unsafe symlink component")


def _canonical(value: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    parent = _resolve_existing_private_directory(raw.parent)
    return parent / raw.name


def _require_exact_path(value: str | Path, expected: str | Path, field: str) -> Path:
    observed = _canonical(value)
    canonical_expected = _canonical(expected)
    if observed != canonical_expected:
        raise ValueError(f"{field} is not the canonical private location")
    return observed


def _normalized_code_constant(
    value: object,
    *,
    _depth: int,
    _code_count: list[int],
    _constant_count: list[int],
) -> object:
    if _depth > _MAX_LIVE_SEMANTIC_DEPTH:
        raise ValueError("live semantic code constant exceeds its depth bound")
    _constant_count[0] += 1
    if _constant_count[0] > _MAX_LIVE_SEMANTIC_NODES:
        raise ValueError("live semantic code constants exceed their node bound")
    if isinstance(value, CodeType):
        return _normalized_code(
            value,
            _depth=_depth,
            _count=_code_count,
            _constant_count=_constant_count,
        )
    if value is None:
        return (b"none",)
    if value is Ellipsis:
        return (b"ellipsis",)
    if type(value) is bool:
        return (b"bool", b"1" if value else b"0")
    if type(value) is int:
        if value.bit_length() > _MAX_LIVE_SEMANTIC_LEAF_BYTES * 8:
            raise ValueError("live semantic integer constant exceeds its byte bound")
        return (b"int", str(value).encode("ascii"))
    if type(value) is float:
        return (b"float", value.hex().encode("ascii"))
    if type(value) is complex:
        return (
            b"complex",
            value.real.hex().encode("ascii"),
            value.imag.hex().encode("ascii"),
        )
    if type(value) is str:
        if len(value) > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
            raise ValueError("live semantic string constant exceeds its byte bound")
        payload = value.encode("utf-8")
        if len(payload) > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
            raise ValueError("live semantic string constant exceeds its byte bound")
        return (b"str", payload)
    if type(value) is bytes:
        if len(value) > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
            raise ValueError("live semantic bytes constant exceeds its byte bound")
        return (b"bytes", memoryview(value).tobytes())
    if type(value) is tuple:
        if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
            raise ValueError("live semantic tuple constant exceeds its item bound")
        return (
            b"tuple",
            tuple(
                _normalized_code_constant(
                    item,
                    _depth=_depth + 1,
                    _code_count=_code_count,
                    _constant_count=_constant_count,
                )
                for item in value
            ),
        )
    if type(value) is frozenset:
        if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
            raise ValueError("live semantic frozenset constant exceeds its item bound")
        items = [
            _normalized_code_constant(
                item,
                _depth=_depth + 1,
                _code_count=_code_count,
                _constant_count=_constant_count,
            )
            for item in value
        ]
        items.sort(key=marshal.dumps)
        return (b"frozenset", tuple(items))
    raise ValueError(
        "unsupported live semantic code constant: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _normalized_code(
    value: CodeType,
    *,
    _depth: int = 0,
    _count: list[int] | None = None,
    _constant_count: list[int] | None = None,
) -> CodeType:
    """Rebuild code into a canonical, execution-state-free marshal form."""
    if not isinstance(value, CodeType):
        raise ValueError("live semantic code object is malformed")
    if _depth > _MAX_LIVE_SEMANTIC_DEPTH:
        raise ValueError("live semantic code exceeds its depth bound")
    if len(value.co_consts) > _MAX_LIVE_SEMANTIC_CONTAINER:
        raise ValueError("live semantic code exceeds its constant bound")
    if _count is None:
        _count = [0]
    if _constant_count is None:
        _constant_count = [0]
    _count[0] += 1
    if _count[0] > _MAX_LIVE_PRODUCER_COMPONENTS:
        raise ValueError("live semantic code exceeds its component bound")
    constants = tuple(
        _normalized_code_constant(
            item,
            _depth=_depth + 1,
            _code_count=_count,
            _constant_count=_constant_count,
        )
        for item in value.co_consts
    )
    # CodeType.replace() on an already executed code object preserves hidden
    # Python 3.9 opcache bytes.  Start from a never-executed template and copy
    # every public semantic field so the digest cannot depend on warm-up order.
    return _EMPTY_CODE_TEMPLATE.replace(
        co_argcount=value.co_argcount,
        co_posonlyargcount=value.co_posonlyargcount,
        co_kwonlyargcount=value.co_kwonlyargcount,
        co_nlocals=value.co_nlocals,
        co_stacksize=value.co_stacksize,
        co_flags=value.co_flags,
        co_firstlineno=value.co_firstlineno,
        co_code=memoryview(value.co_code).tobytes(),
        co_consts=constants,
        co_names=tuple(sys.intern(item) for item in value.co_names),
        co_varnames=tuple(sys.intern(item) for item in value.co_varnames),
        co_freevars=tuple(sys.intern(item) for item in value.co_freevars),
        co_cellvars=tuple(sys.intern(item) for item in value.co_cellvars),
        co_filename=sys.intern(""),
        co_name=sys.intern(value.co_name),
        co_lnotab=memoryview(value.co_lnotab).tobytes(),
    )


def _code_dispatch_bindings(
    value: CodeType,
    *,
    recursive: bool,
    opnames: frozenset[str],
) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
    """Return exact bytecode roots and their direct static attribute paths."""
    if not isinstance(value, CodeType) or not opnames:
        raise ValueError("producer dispatch query is malformed")
    pending = [value]
    visited: set[int] = set()
    bindings: dict[str, set[tuple[str, ...]]] = {}
    instruction_count = 0
    while pending:
        code = pending.pop()
        identity = id(code)
        if identity in visited:
            continue
        visited.add(identity)
        if len(visited) > _MAX_LIVE_PRODUCER_COMPONENTS:
            raise ValueError("nested producer code exceeds its component bound")
        instructions = tuple(dis.get_instructions(code))
        instruction_count += len(instructions)
        if instruction_count > _MAX_LIVE_SEMANTIC_NODES:
            raise ValueError("producer bytecode exceeds its instruction bound")
        for index, instruction in enumerate(instructions):
            if instruction.opname not in opnames:
                continue
            name = instruction.argval
            if type(name) is not str or not name:
                raise ValueError("producer dispatch root is malformed")
            path: list[str] = []
            for following in instructions[index + 1:]:
                if following.opname in {"CACHE", "EXTENDED_ARG"}:
                    continue
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                segment = following.argval
                if type(segment) is not str or not segment:
                    raise ValueError("producer dispatch segment is malformed")
                path.append(segment)
                if len(path) > _MAX_LIVE_SEMANTIC_DEPTH:
                    raise ValueError("producer dispatch path exceeds its depth bound")
            bindings.setdefault(name, set()).add(tuple(path))
            if (
                len(bindings) > _MAX_LIVE_SEMANTIC_CONTAINER
                or sum(len(paths) for paths in bindings.values())
                > _MAX_LIVE_SEMANTIC_CONTAINER
            ):
                raise ValueError("producer dispatch closure exceeds its item bound")
        semantic_instructions = tuple(
            item for item in instructions
            if item.opname not in {"CACHE", "EXTENDED_ARG", "PRECALL"}
        )
        for index, instruction in enumerate(semantic_instructions):
            if (
                instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}
                or instruction.argval != "getattr"
                or index + 3 >= len(semantic_instructions)
            ):
                continue
            target = semantic_instructions[index + 1]
            if target.opname not in opnames:
                continue
            target_name = target.argval
            if type(target_name) is not str or not target_name:
                raise ValueError("literal getattr dispatch root is malformed")
            cursor = index + 2
            path: list[str] = []
            while (
                cursor < len(semantic_instructions)
                and semantic_instructions[cursor].opname
                in {"LOAD_ATTR", "LOAD_METHOD"}
            ):
                segment = semantic_instructions[cursor].argval
                if type(segment) is not str or not segment:
                    raise ValueError("literal getattr dispatch path is malformed")
                path.append(segment)
                cursor += 1
            if cursor >= len(semantic_instructions):
                continue
            attribute = semantic_instructions[cursor]
            if attribute.opname != "LOAD_CONST" or type(attribute.argval) is not str:
                continue
            path.append(attribute.argval)
            if len(path) > _MAX_LIVE_SEMANTIC_DEPTH:
                raise ValueError("literal getattr dispatch exceeds its depth bound")
            tail = semantic_instructions[cursor + 1:cursor + 3]
            call = None
            if tail and tail[0].opname in {"CALL", "CALL_FUNCTION"}:
                call = tail[0]
                expected_arguments = 2
            elif (
                len(tail) == 2
                and tail[0].opname == "LOAD_CONST"
                and tail[1].opname in {"CALL", "CALL_FUNCTION"}
            ):
                call = tail[1]
                expected_arguments = 3
            else:
                continue
            if call.arg != expected_arguments:
                continue
            bindings.setdefault(target_name, set()).add(tuple(path))
            if (
                len(bindings) > _MAX_LIVE_SEMANTIC_CONTAINER
                or sum(len(paths) for paths in bindings.values())
                > _MAX_LIVE_SEMANTIC_CONTAINER
            ):
                raise ValueError("producer dispatch closure exceeds its item bound")
        for index, instruction in enumerate(semantic_instructions):
            if instruction.opname not in opnames:
                continue
            root_name = instruction.argval
            if type(root_name) is not str or not root_name:
                raise ValueError("producer call dispatch root is malformed")
            cursor = index + 1
            direct_path: list[str] = []
            while (
                cursor < len(semantic_instructions)
                and semantic_instructions[cursor].opname
                in {"LOAD_ATTR", "LOAD_METHOD"}
            ):
                segment = semantic_instructions[cursor].argval
                if type(segment) is not str or not segment:
                    raise ValueError("producer call dispatch path is malformed")
                direct_path.append(segment)
                cursor += 1
            relative_depth = 1
            call_index: int | None = None
            scan_limit = min(
                len(semantic_instructions),
                cursor + (_MAX_LIVE_SEMANTIC_DEPTH * 4),
            )
            for candidate_index in range(cursor, scan_limit):
                candidate = semantic_instructions[candidate_index]
                if (
                    candidate.opname in {"CALL", "CALL_FUNCTION"}
                    and type(candidate.arg) is int
                    and relative_depth == candidate.arg + 1
                ):
                    call_index = candidate_index
                    break
                if (
                    "JUMP" in candidate.opname
                    or candidate.opname in {
                        "FOR_ITER", "RAISE_VARARGS", "RETURN_VALUE",
                        "SETUP_FINALLY", "YIELD_FROM", "YIELD_VALUE",
                    }
                ):
                    break
                try:
                    effect = dis.stack_effect(candidate.opcode, candidate.arg)
                except ValueError:
                    effect = dis.stack_effect(candidate.opcode)
                relative_depth += effect
                if relative_depth <= 0:
                    break
            if call_index is None:
                continue
            result_path: list[str] = []
            cursor = call_index + 1
            while (
                cursor < len(semantic_instructions)
                and semantic_instructions[cursor].opname
                in {"LOAD_ATTR", "LOAD_METHOD"}
            ):
                segment = semantic_instructions[cursor].argval
                if type(segment) is not str or not segment:
                    raise ValueError("producer call-result dispatch path is malformed")
                result_path.append(segment)
                cursor += 1
            path = tuple(
                (*direct_path, _LIVE_CALL_RESULT_SEGMENT, *result_path)
            )
            if len(path) > _MAX_LIVE_SEMANTIC_DEPTH:
                raise ValueError("producer call-result dispatch exceeds its depth bound")
            bindings.setdefault(root_name, set()).add(path)
            if (
                len(bindings) > _MAX_LIVE_SEMANTIC_CONTAINER
                or sum(len(paths) for paths in bindings.values())
                > _MAX_LIVE_SEMANTIC_CONTAINER
            ):
                raise ValueError("producer dispatch closure exceeds its item bound")
        if recursive:
            nested = [item for item in code.co_consts if isinstance(item, CodeType)]
            if len(nested) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("nested producer code exceeds its item bound")
            pending.extend(nested)
    return tuple(
        (name, tuple(sorted(paths)))
        for name, paths in sorted(bindings.items())
    )


def _normalize_dispatch_paths(
    value: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    paths: set[tuple[str, ...]] = set()
    for raw_path in value:
        path = tuple(raw_path)
        if (
            len(path) > _MAX_LIVE_SEMANTIC_DEPTH
            or any(type(segment) is not str or not segment for segment in path)
        ):
            raise ValueError("live dispatch path is malformed or oversized")
        paths.add(path)
    if len(paths) > _MAX_LIVE_SEMANTIC_CONTAINER:
        raise ValueError("live dispatch path set exceeds its item bound")
    return tuple(sorted(paths))


class _LiveSemanticEncoder:
    """Strict, bounded canonical encoder for live producer behavior."""

    def __init__(self, producer_modules: Sequence[str]):
        self._producer_modules = frozenset(producer_modules)
        self._memo: dict[tuple[object, ...], int] = {}
        self._memo_values: dict[tuple[object, ...], object] = {}
        self._next_ordinal = 0
        self._node_count = 0
        self._total_bytes = 0

    def _frame(self, tag: bytes, payload: bytes = b"") -> bytes:
        if type(tag) is not bytes or not tag:
            raise RuntimeError("live semantic tag is malformed")
        if len(payload) > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
            raise ValueError("live semantic leaf exceeds its byte bound")
        framed = len(tag).to_bytes(2, "big") + tag + len(payload).to_bytes(8, "big") + payload
        self._total_bytes += len(framed)
        if self._total_bytes > _MAX_LIVE_SEMANTIC_TOTAL_BYTES:
            raise ValueError("live semantic closure exceeds its byte bound")
        return framed

    def _enter(
        self,
        value: object,
        depth: int,
        *,
        context: tuple[object, ...] = ("generic",),
    ) -> tuple[bytes | None, int]:
        if depth > _MAX_LIVE_SEMANTIC_DEPTH:
            raise ValueError("live semantic closure exceeds its depth bound")
        self._node_count += 1
        if self._node_count > _MAX_LIVE_SEMANTIC_NODES:
            raise ValueError("live semantic closure exceeds its node bound")
        identity = (id(value), *context)
        if identity in self._memo:
            return self._frame(b"ref", self._memo[identity].to_bytes(8, "big")), -1
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._memo[identity] = ordinal
        self._memo_values[identity] = value
        return None, ordinal

    @staticmethod
    def _sort_token(value: object, depth: int = 0) -> bytes:
        if depth > _MAX_LIVE_SEMANTIC_DEPTH:
            raise ValueError("live semantic set ordering exceeds its depth bound")
        if value is None:
            return b"n"
        if value is Ellipsis:
            return b"e"
        if type(value) is bool:
            return b"b1" if value else b"b0"
        if type(value) is int:
            estimated_digits = (value.bit_length() * 30103 // 100000) + 2
            if estimated_digits > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                raise ValueError("live semantic integer exceeds its byte bound")
            payload = str(value).encode("ascii")
            if len(payload) + 1 > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                raise ValueError("live semantic integer exceeds its byte bound")
            return b"i" + payload
        if type(value) is float:
            return b"f" + value.hex().encode("ascii")
        if type(value) is complex:
            return (
                b"c" + value.real.hex().encode("ascii") + b"\0"
                + value.imag.hex().encode("ascii")
            )
        if type(value) is str:
            if len(value) + 1 > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                raise ValueError("live semantic string exceeds its byte bound")
            byte_length = 1
            for character in value:
                ordinal = ord(character)
                byte_length += (
                    1 if ordinal < 0x80 else 2 if ordinal < 0x800
                    else 3 if ordinal < 0x10000 else 4
                )
                if byte_length > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                    raise ValueError("live semantic string exceeds its byte bound")
            return b"s" + value.encode("utf-8")
        if type(value) is bytes:
            if len(value) + 1 > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                raise ValueError("live semantic bytes exceed their byte bound")
            return b"y" + value
        if type(value) is tuple:
            if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("live semantic tuple exceeds its item bound")
            items: list[bytes] = []
            byte_length = 1
            for child in value:
                item = _LiveSemanticEncoder._sort_token(child, depth + 1)
                byte_length += 8 + len(item)
                if byte_length > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                    raise ValueError("live semantic tuple exceeds its byte bound")
                items.append(item)
            return b"t" + b"".join(
                len(item).to_bytes(8, "big") + item for item in items
            )
        if type(value) is frozenset:
            if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("live semantic frozenset exceeds its item bound")
            items = []
            byte_length = 1
            for child in value:
                item = _LiveSemanticEncoder._sort_token(child, depth + 1)
                byte_length += 8 + len(item)
                if byte_length > _MAX_LIVE_SEMANTIC_LEAF_BYTES:
                    raise ValueError("live semantic frozenset exceeds its byte bound")
                items.append(item)
            items.sort()
            return b"r" + b"".join(
                len(item).to_bytes(8, "big") + item for item in items
            )
        raise ValueError("live semantic set contains an unsupported value")

    def _encode_sequence(self, tag: bytes, value: object, depth: int) -> bytes:
        repeated, ordinal = self._enter(value, depth)
        if repeated is not None:
            return repeated
        items = tuple(value)  # type: ignore[arg-type]
        if len(items) > _MAX_LIVE_SEMANTIC_CONTAINER:
            raise ValueError("live semantic container exceeds its item bound")
        payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
        payload += self._frame(b"length", len(items).to_bytes(8, "big"))
        for item in items:
            payload += self.encode(item, depth=depth + 1)
        return self._frame(tag, payload)

    def _encode_mapping(self, value: Mapping[object, object], depth: int) -> bytes:
        repeated, ordinal = self._enter(value, depth)
        if repeated is not None:
            return repeated
        if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
            raise ValueError("live semantic mapping exceeds its item bound")
        if any(type(key) is not str for key in value):
            raise ValueError("live semantic mappings require exact string keys")
        payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
        payload += self._frame(b"length", len(value).to_bytes(8, "big"))
        for key in sorted(value):  # type: ignore[type-var]
            payload += self.encode(key, depth=depth + 1)
            payload += self.encode(value[key], depth=depth + 1)
        return self._frame(b"mapping", payload)

    def _encode_function(
        self,
        value: Callable[..., object],
        depth: int,
        *,
        bind_external_behavior: bool = False,
    ) -> bytes:
        code = getattr(value, "__code__", None)
        if not isinstance(code, CodeType):
            raise ValueError("live Python function has no canonical code object")
        module_name = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if type(module_name) is not str or type(qualname) is not str:
            raise ValueError("live Python function identity is malformed")
        bind_full_behavior = (
            module_name in self._producer_modules or bind_external_behavior
        )
        repeated, ordinal = self._enter(
            value,
            depth,
            context=(
                "python-function",
                "full-behavior" if bind_full_behavior else "shallow-identity",
            ),
        )
        if repeated is not None:
            return repeated
        payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
        payload += self.encode(module_name, depth=depth + 1)
        payload += self.encode(qualname, depth=depth + 1)
        payload += self.encode(code, depth=depth + 1)
        # Ordinary external callables remain bounded imported identities.  An
        # external method reachable from a constructed type is behavior-bearing
        # and therefore binds its defaults, closure, and referenced globals.
        if not bind_full_behavior:
            return self._frame(b"external-function", payload)
        payload += self.encode(getattr(value, "__defaults__", None), depth=depth + 1)
        payload += self.encode(getattr(value, "__kwdefaults__", None), depth=depth + 1)

        global_dispatch = _code_dispatch_bindings(
            code,
            recursive=True,
            opnames=frozenset({"LOAD_GLOBAL", "LOAD_NAME"}),
        )
        closure_dispatch = dict(_code_dispatch_bindings(
            code,
            recursive=True,
            opnames=frozenset({"LOAD_DEREF", "LOAD_CLASSDEREF"}),
        ))

        freevars = tuple(code.co_freevars)
        closure = getattr(value, "__closure__", None)
        closure_cells = () if closure is None else tuple(closure)
        if len(freevars) != len(closure_cells):
            raise ValueError("live Python function closure is malformed")
        closure_payload = self._frame(b"length", len(freevars).to_bytes(8, "big"))
        for name, cell in zip(freevars, closure_cells):
            closure_payload += self.encode(name, depth=depth + 1)
            try:
                cell_value = cell.cell_contents
            except ValueError:
                closure_payload += self._frame(b"empty-cell")
            else:
                closure_payload += self.encode(
                    cell_value,
                    depth=depth + 1,
                    dispatch_paths=closure_dispatch.get(name, ()),
                )
        payload += self._frame(b"closure", closure_payload)

        globals_map = getattr(value, "__globals__", None)
        if type(globals_map) is not dict:
            raise ValueError("live Python function globals are malformed")
        builtins_value = getattr(value, "__builtins__", {})
        if isinstance(builtins_value, ModuleType):
            builtins_map = vars(builtins_value)
        elif type(builtins_value) is dict:
            builtins_map = builtins_value
        else:
            raise ValueError("live Python function builtins are malformed")
        bindings = bytearray()
        binding_count = 0
        for name, dispatch_paths in global_dispatch:
            source: str | None = None
            bound: object
            if name in globals_map:
                source, bound = "global", globals_map[name]
            elif name in builtins_map:
                source, bound = "builtin", builtins_map[name]
            else:
                continue
            bindings.extend(self.encode(source, depth=depth + 1))
            bindings.extend(self.encode(name, depth=depth + 1))
            bindings.extend(self.encode(
                bound,
                depth=depth + 1,
                dispatch_paths=dispatch_paths,
            ))
            binding_count += 1
        payload += self._frame(
            b"bindings",
            self._frame(b"length", binding_count.to_bytes(8, "big")) + bytes(bindings),
        )
        return self._frame(b"function", payload)

    def _encode_module(
        self,
        value: ModuleType,
        depth: int,
        dispatch_paths: Sequence[Sequence[str]],
    ) -> bytes:
        normalized_paths = _normalize_dispatch_paths(dispatch_paths)
        repeated, ordinal = self._enter(
            value,
            depth,
            context=("module", normalized_paths),
        )
        if repeated is not None:
            return repeated
        module_name = getattr(value, "__name__", None)
        if type(module_name) is not str:
            raise ValueError("live module identity is malformed")
        payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
        payload += self.encode(module_name, depth=depth + 1)
        namespace = vars(value)
        grouped: dict[str, set[tuple[str, ...]]] = {}
        for path in normalized_paths:
            if not path:
                continue
            grouped.setdefault(path[0], set()).add(path[1:])
        selected: list[tuple[str, object, tuple[tuple[str, ...], ...]]] = []
        for name, tails in sorted(grouped.items()):
            if name not in namespace or name.startswith("__"):
                raise ValueError("live module dispatch attribute is unavailable")
            bound = namespace[name]
            if value is sys and name == "modules":
                bound = {
                    producer_name: getattr(sys.modules.get(producer_name), "__name__", None)
                    for producer_name in sorted(self._producer_modules)
                }
            selected.append((name, bound, tuple(sorted(tails))))
        if len(selected) > _MAX_LIVE_SEMANTIC_CONTAINER:
            raise ValueError("live module dispatch closure exceeds its item bound")
        attributes = self._frame(b"length", len(selected).to_bytes(8, "big"))
        for name, bound, tails in selected:
            attributes += self.encode(name, depth=depth + 1)
            attributes += self.encode(
                bound,
                depth=depth + 1,
                dispatch_paths=tails,
            )
        payload += self._frame(b"attributes", attributes)
        return self._frame(b"module", payload)

    def encode(
        self,
        value: object,
        *,
        depth: int = 0,
        dispatch_paths: Sequence[Sequence[str]] = (),
    ) -> bytes:
        if depth > _MAX_LIVE_SEMANTIC_DEPTH:
            raise ValueError("live semantic closure exceeds its depth bound")
        self._node_count += 1
        if self._node_count > _MAX_LIVE_SEMANTIC_NODES:
            raise ValueError("live semantic closure exceeds its node bound")
        if value is None:
            return self._frame(b"none")
        if _is_original_trainer_authorization_marker(value):
            return self._frame(
                b"dynamic-landmark-ssl-authorization-marker"
            )
        if value is Ellipsis:
            return self._frame(b"ellipsis")
        if type(value) is bool:
            return self._frame(b"bool", b"1" if value else b"0")
        if type(value) is int:
            return self._frame(b"int", str(value).encode("ascii"))
        if type(value) is float:
            return self._frame(b"float", value.hex().encode("ascii"))
        if type(value) is complex:
            return self._frame(
                b"complex",
                value.real.hex().encode("ascii") + b"\0"
                + value.imag.hex().encode("ascii"),
            )
        if type(value) is str:
            return self._frame(b"str", value.encode("utf-8"))
        if type(value) is bytes:
            return self._frame(b"bytes", value)
        if isinstance(value, enum.Enum):
            repeated, ordinal = self._enter(value, depth)
            if repeated is not None:
                return repeated
            name = value.name
            if name is not None and type(name) is not str:
                raise ValueError("live enum name is malformed")
            return self._frame(
                b"enum",
                self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
                + self.encode(type(value), depth=depth + 1)
                + self.encode(name, depth=depth + 1)
                + self.encode(value.value, depth=depth + 1),
            )
        if isinstance(value, Path):
            return self._frame(b"path", Path(value).as_posix().encode("utf-8"))
        if isinstance(value, CodeType):
            payload = marshal.dumps(_normalized_code(value))
            return self._frame(b"code", payload)
        if type(value) is tuple:
            return self._encode_sequence(b"tuple", value, depth)
        if type(value) is list:
            return self._encode_sequence(b"list", value, depth)
        if type(value) in {set, frozenset}:
            repeated, ordinal = self._enter(value, depth)
            if repeated is not None:
                return repeated
            if len(value) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("live semantic set exceeds its item bound")
            ordered = sorted(value, key=self._sort_token)
            payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
            payload += self._frame(b"length", len(ordered).to_bytes(8, "big"))
            for item in ordered:
                payload += self.encode(item, depth=depth + 1)
            return self._frame(
                b"set" if type(value) is set else b"frozenset",
                payload,
            )
        if type(value) is dict:
            return self._encode_mapping(value, depth)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            repeated, ordinal = self._enter(value, depth)
            if repeated is not None:
                return repeated
            payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
            payload += self.encode(type(value), depth=depth + 1)
            fields = tuple(dataclasses.fields(value))
            if len(fields) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("live dataclass exceeds its field bound")
            payload += self._frame(b"length", len(fields).to_bytes(8, "big"))
            for item in fields:
                payload += self.encode(item.name, depth=depth + 1)
                payload += self.encode(getattr(value, item.name), depth=depth + 1)
            return self._frame(b"dataclass", payload)
        if isinstance(value, staticmethod):
            return self._frame(
                b"staticmethod",
                self.encode(
                    value.__func__, depth=depth + 1,
                    dispatch_paths=dispatch_paths,
                ),
            )
        if isinstance(value, classmethod):
            return self._frame(
                b"classmethod",
                self.encode(
                    value.__func__, depth=depth + 1,
                    dispatch_paths=dispatch_paths,
                ),
            )
        if isinstance(value, property):
            value_type = type(value)
            module_name = getattr(value_type, "__module__", None)
            qualname = getattr(value_type, "__qualname__", None)
            if type(module_name) is not str or type(qualname) is not str:
                raise ValueError("live property descriptor identity is malformed")
            normalized_paths = _normalize_dispatch_paths(dispatch_paths)
            bind_full_behavior = (
                (_LIVE_EXTERNAL_BEHAVIOR_SEGMENT,) in normalized_paths
                or any(
                    getattr(accessor, "__module__", None)
                    in self._producer_modules
                    for accessor in (value.fget, value.fset, value.fdel)
                    if accessor is not None
                )
            )
            repeated, ordinal = self._enter(
                value,
                depth,
                context=(
                    "property",
                    "full-behavior"
                    if bind_full_behavior
                    else "shallow-identity",
                ),
            )
            if repeated is not None:
                return repeated
            return self._frame(
                b"property",
                self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
                + self.encode(module_name, depth=depth + 1)
                + self.encode(qualname, depth=depth + 1)
                + self.encode(
                    value.fget, depth=depth + 1,
                    dispatch_paths=dispatch_paths,
                )
                + self.encode(
                    value.fset, depth=depth + 1,
                    dispatch_paths=dispatch_paths,
                )
                + self.encode(
                    value.fdel, depth=depth + 1,
                    dispatch_paths=dispatch_paths,
                ),
            )
        if type(value) in {operator.attrgetter, operator.itemgetter}:
            reduced = value.__reduce__()
            if (
                type(reduced) is not tuple
                or len(reduced) != 2
                or reduced[0] is not type(value)
                or type(reduced[1]) is not tuple
                or len(reduced[1]) > _MAX_LIVE_SEMANTIC_CONTAINER
            ):
                raise ValueError("live operator callable identity is malformed")
            return self._frame(
                b"operator-callable",
                self.encode(type(value).__module__, depth=depth + 1)
                + self.encode(type(value).__qualname__, depth=depth + 1)
                + self.encode(reduced[1], depth=depth + 1),
            )
        if inspect.isfunction(value):
            normalized_paths = _normalize_dispatch_paths(dispatch_paths)
            return self._encode_function(
                value,
                depth,
                bind_external_behavior=(
                    (_LIVE_EXTERNAL_BEHAVIOR_SEGMENT,) in normalized_paths
                ),
            )
        if inspect.isbuiltin(value):
            module_name = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
            value_type = type(value)
            if (
                module_name is not None and type(module_name) is not str
                or type(qualname) is not str
                or type(value_type.__module__) is not str
                or type(value_type.__qualname__) is not str
            ):
                raise ValueError("live builtin identity is malformed")
            return self._frame(
                b"builtin",
                self.encode(value_type.__module__, depth=depth + 1)
                + self.encode(value_type.__qualname__, depth=depth + 1)
                + self.encode(module_name, depth=depth + 1)
                + self.encode(qualname, depth=depth + 1),
            )
        if inspect.ismethoddescriptor(value):
            owner = getattr(value, "__objclass__", None)
            name = getattr(value, "__name__", None)
            if isinstance(owner, type) and type(name) is str:
                return self._frame(
                    b"method-descriptor",
                    self.encode(owner, depth=depth + 1)
                    + self.encode(name, depth=depth + 1),
                )
        if inspect.isgetsetdescriptor(value) or inspect.ismemberdescriptor(value):
            owner = getattr(value, "__objclass__", None)
            name = getattr(value, "__name__", None)
            if not isinstance(owner, type) or type(name) is not str:
                raise ValueError("live data descriptor identity is malformed")
            return self._frame(
                b"getset-descriptor" if inspect.isgetsetdescriptor(value)
                else b"member-descriptor",
                self.encode(owner, depth=depth + 1)
                + self.encode(name, depth=depth + 1),
            )
        numpy_module = sys.modules.get("numpy")
        numpy_ufunc_type = (
            getattr(numpy_module, "ufunc", None)
            if isinstance(numpy_module, ModuleType)
            else None
        )
        if (
            isinstance(numpy_module, ModuleType)
            and value is getattr(numpy_module, "_NoValue", object())
        ):
            return self._frame(b"numpy-no-value-sentinel")
        if isinstance(numpy_ufunc_type, type) and type(value) is numpy_ufunc_type:
            name = getattr(value, "__name__", None)
            nin = getattr(value, "nin", None)
            nout = getattr(value, "nout", None)
            nargs = getattr(value, "nargs", None)
            ntypes = getattr(value, "ntypes", None)
            raw_types = getattr(value, "types", None)
            identity = getattr(value, "identity", None)
            signature = getattr(value, "signature", None)
            if (
                type(name) is not str
                or type(nin) is not int
                or type(nout) is not int
                or type(nargs) is not int
                or type(ntypes) is not int
                or type(raw_types) is not list
                or any(type(item) is not str for item in raw_types)
                or len(raw_types) != ntypes
                or len(raw_types) > _MAX_LIVE_SEMANTIC_CONTAINER
                or nin < 0
                or nout < 0
                or nargs != nin + nout
                or ntypes < 0
                or (signature is not None and type(signature) is not str)
            ):
                raise ValueError("live NumPy ufunc identity is malformed")
            return self._frame(
                b"numpy-ufunc",
                self.encode(type(value), depth=depth + 1)
                + self.encode(name, depth=depth + 1)
                + self.encode(nin, depth=depth + 1)
                + self.encode(nout, depth=depth + 1)
                + self.encode(nargs, depth=depth + 1)
                + self.encode(ntypes, depth=depth + 1)
                + self.encode(tuple(raw_types), depth=depth + 1)
                + self.encode(identity, depth=depth + 1)
                + self.encode(signature, depth=depth + 1),
            )
        torch_module = sys.modules.get("torch")
        torch_dtype_type = (
            getattr(torch_module, "dtype", None)
            if isinstance(torch_module, ModuleType)
            else None
        )
        if isinstance(torch_dtype_type, type) and type(value) is torch_dtype_type:
            identity = str(value)
            if not identity.startswith("torch.") or not identity[6:].isalnum():
                raise ValueError("live torch dtype identity is malformed")
            return self._frame(
                b"torch-dtype", self.encode(identity, depth=depth + 1)
            )
        value_type = type(value)
        if (
            isinstance(numpy_module, ModuleType)
            and value_type.__module__ == "numpy"
            and value_type.__qualname__ == "_ArrayFunctionDispatcher"
        ):
            namespace = vars(value)
            module_name = namespace.get("__module__")
            name = namespace.get("__name__")
            qualname = namespace.get("__qualname__")
            wrapped = namespace.get("__wrapped__")
            if (
                type(module_name) is not str
                or type(name) is not str
                or type(qualname) is not str
                or not (inspect.isfunction(wrapped) or inspect.isbuiltin(wrapped))
            ):
                raise ValueError("live NumPy dispatcher identity is malformed")
            return self._frame(
                b"numpy-array-function-dispatcher",
                self.encode(module_name, depth=depth + 1)
                + self.encode(name, depth=depth + 1)
                + self.encode(qualname, depth=depth + 1)
                + self.encode(wrapped, depth=depth + 1),
            )
        instance_namespace = getattr(value, "__dict__", None)
        if (
            type(instance_namespace) is dict
            and not isinstance(value, ModuleType)
            and type(value).__module__ not in self._producer_modules
        ):
            repeated, ordinal = self._enter(
                value, depth, context=("external-instance",)
            )
            if repeated is not None:
                return repeated
            if (
                len(instance_namespace) > _MAX_LIVE_SEMANTIC_CONTAINER
                or any(type(name) is not str for name in instance_namespace)
            ):
                raise ValueError(
                    "live external instance namespace is malformed or oversized"
                )
            return self._frame(
                b"external-instance",
                self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
                + self.encode(
                    type(value),
                    depth=depth + 1,
                    dispatch_paths=((_LIVE_CALL_RESULT_SEGMENT,),),
                )
                + self.encode(instance_namespace, depth=depth + 1),
            )
        if isinstance(value, ModuleType):
            return self._encode_module(value, depth, dispatch_paths)
        if isinstance(value, type):
            module_name = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", None)
            if type(module_name) is not str or type(qualname) is not str:
                raise ValueError("live type identity is malformed")
            producer_type = module_name in self._producer_modules
            normalized_paths = _normalize_dispatch_paths(dispatch_paths)
            context = (
                ("producer-type-full",)
                if producer_type
                else ("external-type", normalized_paths)
            )
            repeated, ordinal = self._enter(value, depth, context=context)
            if repeated is not None:
                return repeated
            payload = self._frame(b"ordinal", ordinal.to_bytes(8, "big"))
            payload += self.encode(module_name, depth=depth + 1)
            payload += self.encode(qualname, depth=depth + 1)
            bases = tuple(
                (getattr(base, "__module__", None), getattr(base, "__qualname__", None))
                for base in value.__bases__
            )
            if any(type(module) is not str or type(name) is not str
                   for module, name in bases):
                raise ValueError("live type base identity is malformed")
            payload += self.encode(bases, depth=depth + 1)
            metaclass = type(value)
            payload += self.encode(
                (getattr(metaclass, "__module__", None),
                 getattr(metaclass, "__qualname__", None)),
                depth=depth + 1,
            )
            selected: list[tuple[str, object, tuple[tuple[str, ...], ...]]] = []
            if producer_type:
                skipped = {
                    "__annotations__", "__dataclass_fields__", "__dataclass_params__",
                    "__dict__", "__doc__", "__module__", "__weakref__",
                }
                for name, raw in sorted(vars(value).items()):
                    if name not in skipped:
                        selected.append((name, raw, ()))
            else:
                grouped: dict[str, set[tuple[str, ...]]] = {}
                call_result_paths: set[tuple[str, ...]] = set()
                for path in normalized_paths:
                    if path and path[0] in {
                        _LIVE_CALL_RESULT_SEGMENT,
                        _LIVE_EXTERNAL_BEHAVIOR_SEGMENT,
                    }:
                        call_result_paths.add(path[1:])
                    elif path:
                        grouped.setdefault(path[0], set()).add(path[1:])
                if call_result_paths:
                    grouped.setdefault("__new__", set()).add(())
                    grouped.setdefault("__init__", set()).add(())
                    for path in call_result_paths:
                        if path:
                            grouped.setdefault(path[0], set()).add(path[1:])
                    try:
                        metaclass_call = inspect.getattr_static(
                            metaclass, "__call__"
                        )
                    except AttributeError as exc:
                        raise ValueError(
                            "live external type metaclass call is unavailable"
                        ) from exc
                    payload += self._frame(
                        b"metaclass-call",
                        self.encode(metaclass_call, depth=depth + 1),
                    )
                    behavior_members: list[tuple[str, str, str, object]] = []
                    mro = tuple(value.__mro__)
                    if len(mro) > _MAX_LIVE_SEMANTIC_DEPTH:
                        raise ValueError(
                            "live external type MRO exceeds its depth bound"
                        )
                    for owner in mro:
                        owner_module = getattr(owner, "__module__", None)
                        owner_qualname = getattr(owner, "__qualname__", None)
                        if (
                            type(owner_module) is not str
                            or type(owner_qualname) is not str
                        ):
                            raise ValueError(
                                "live external type MRO identity is malformed"
                            )
                        for name, raw in sorted(vars(owner).items()):
                            if (
                                isinstance(raw, (staticmethod, classmethod, property))
                                or inspect.isfunction(raw)
                                or inspect.isbuiltin(raw)
                                or inspect.ismethoddescriptor(raw)
                                or inspect.isgetsetdescriptor(raw)
                                or inspect.ismemberdescriptor(raw)
                            ):
                                behavior_members.append((
                                    owner_module, owner_qualname, name, raw,
                                ))
                                if (
                                    len(behavior_members)
                                    > _MAX_LIVE_SEMANTIC_CONTAINER
                                ):
                                    raise ValueError(
                                        "live external type behavior exceeds its item bound"
                                    )
                    behavior_payload = self._frame(
                        b"length", len(behavior_members).to_bytes(8, "big")
                    )
                    for owner_module, owner_qualname, name, raw in behavior_members:
                        behavior_payload += self.encode(
                            (owner_module, owner_qualname), depth=depth + 1
                        )
                        behavior_payload += self.encode(name, depth=depth + 1)
                        behavior_payload += self.encode(
                            raw,
                            depth=depth + 1,
                            dispatch_paths=((_LIVE_EXTERNAL_BEHAVIOR_SEGMENT,),),
                        )
                    payload += self._frame(
                        b"call-result-type-behavior", behavior_payload
                    )
                for name, tails in sorted(grouped.items()):
                    try:
                        raw = inspect.getattr_static(value, name)
                    except AttributeError as exc:
                        raise ValueError(
                            "live type dispatch attribute is unavailable"
                        ) from exc
                    selected.append((name, raw, tuple(sorted(tails))))
            if len(selected) > _MAX_LIVE_SEMANTIC_CONTAINER:
                raise ValueError("live type dispatch exceeds its item bound")
            payload += self._frame(b"length", len(selected).to_bytes(8, "big"))
            for name, raw, tails in selected:
                payload += self.encode(name, depth=depth + 1)
                payload += self.encode(
                    raw,
                    depth=depth + 1,
                    dispatch_paths=tails,
                )
            return self._frame(b"type", payload)
        if isinstance(value, re.Pattern):
            return self._frame(
                b"regex",
                self.encode(value.pattern, depth=depth + 1)
                + self.encode(int(value.flags), depth=depth + 1),
            )
        value_type = type(value)
        if value_type.__module__ == "typing" and type(getattr(value, "_name", None)) is str:
            origin = getattr(value, "__origin__", None)
            return self._frame(
                b"typing-alias",
                self.encode(value._name, depth=depth + 1)
                + self.encode(origin, depth=depth + 1),
            )
        if (
            value_type.__module__ == "dataclasses"
            and value_type.__qualname__ == "_HAS_DEFAULT_FACTORY_CLASS"
        ):
            return self._frame(b"dataclass-default-factory-sentinel")
        raise ValueError(
            "unsupported live semantic value type: "
            f"{value_type.__module__}.{value_type.__qualname__}"
        )


def _live_producer_semantics_sha256() -> str:
    if (
        len(_PRODUCER_MODULE_NAMES) < 1
        or len(_PRODUCER_MODULE_NAMES) > _MAX_LIVE_PRODUCER_MODULES
        or any(type(name) is not str or not name for name in _PRODUCER_MODULE_NAMES)
        or len(set(_PRODUCER_MODULE_NAMES)) != len(_PRODUCER_MODULE_NAMES)
    ):
        raise ValueError("producer module closure is malformed or oversized")
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-live-producer-semantics-v3\0")
    encoder = _LiveSemanticEncoder(_PRODUCER_MODULE_NAMES)
    component_count = 0
    for module_name in _PRODUCER_MODULE_NAMES:
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError("producer module is not live in this process")
        components: list[tuple[str, object]] = []
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module_name:
                components.append((f"function:{name}", value))
            elif inspect.isclass(value) and value.__module__ == module_name:
                components.append((f"class:{name}", value))
                for attribute_name, raw in vars(value).items():
                    candidates: tuple[object, ...]
                    if isinstance(raw, (staticmethod, classmethod)):
                        candidates = (raw.__func__,)
                    elif isinstance(raw, property):
                        candidates = tuple(
                            item for item in (raw.fget, raw.fset, raw.fdel)
                            if item is not None
                        )
                    else:
                        candidates = (raw,)
                    for candidate in candidates:
                        if inspect.isfunction(candidate):
                            components.append((
                                f"class:{name}.{attribute_name}",
                                candidate,
                            ))
        if not components:
            raise RuntimeError("producer module exposes no live executable semantics")
        encoded_module = module_name.encode("utf-8")
        digest.update(len(encoded_module).to_bytes(8, "big"))
        digest.update(encoded_module)
        for logical_name, component in sorted(components, key=lambda item: item[0]):
            component_count += 1
            if component_count > _MAX_LIVE_PRODUCER_COMPONENTS:
                raise ValueError("producer executable closure exceeds its component bound")
            encoded_name = logical_name.encode("utf-8")
            payload = encoder.encode(component)
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    if component_count == 0:
        raise RuntimeError("producer live-semantics closure is empty")
    return digest.hexdigest()


def _producer_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _producer_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-bridge-producer-v3\0")
    live_digest = _live_producer_semantics_sha256().encode("ascii")
    digest.update(live_digest)
    snapshots: list[tuple[Path, int, tuple[int, ...], bytes]] = []
    descriptors = ExitStack()
    aggregate_bytes = 0
    try:
        for path in _PRODUCER_FILES:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            descriptors.callback(os.close, descriptor)
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o022
                or before.st_size < 1
                or before.st_size > _MAX_PRODUCER_FILE_BYTES
                or aggregate_bytes + before.st_size > _MAX_PRODUCER_TOTAL_BYTES
            ):
                raise ValueError("producer source storage is unsafe or oversized")
            payload = bytearray()
            while chunk := os.read(descriptor, 1024 * 1024):
                payload.extend(chunk)
                if len(payload) > _MAX_PRODUCER_FILE_BYTES:
                    raise ValueError("producer source exceeded its file bound")
            after = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            identity = _producer_file_identity(before)
            if (
                len(payload) != before.st_size
                or _producer_file_identity(after) != identity
                or _producer_file_identity(current) != identity
            ):
                raise ValueError("producer source changed during its held snapshot")
            aggregate_bytes += len(payload)
            snapshots.append((path, descriptor, identity, bytes(payload)))
        for index, (path, descriptor, identity, payload) in enumerate(snapshots):
            after = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            if (
                _producer_file_identity(after) != identity
                or _producer_file_identity(current) != identity
            ):
                raise ValueError("producer source set changed during closure")
            logical_name = f"{index}:{path.name}".encode("ascii")
            digest.update(len(logical_name).to_bytes(8, "big"))
            digest.update(logical_name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        final_live_digest = _live_producer_semantics_sha256().encode("ascii")
        if final_live_digest != live_digest:
            raise ValueError("live producer semantics changed during closure")
        for path, descriptor, identity, _payload in snapshots:
            if (
                _producer_file_identity(os.fstat(descriptor)) != identity
                or _producer_file_identity(
                    os.stat(path, follow_symlinks=False)
                ) != identity
            ):
                raise ValueError(
                    "producer source set changed during final live closure"
                )
        return digest.hexdigest()
    finally:
        descriptors.__exit__(*sys.exc_info())


def _key_identity_sha256(path: Path, expected_key: bytes) -> str:
    if type(expected_key) is not bytes or len(expected_key) != 32:
        raise ValueError("authorized canonical key is malformed")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size != 32
        ):
            raise ValueError("canonical key storage is not singly-linked owner-only")
        payload = bytearray()
        while chunk := os.read(descriptor, 64):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_uid, item.st_gid,
            item.st_nlink, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError("canonical key identity changed during authorization")
        if bytes(payload) != expected_key:
            raise ValueError("canonical key bytes contradict live authorization")
        material = json.dumps({
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "mode": int(stat.S_IMODE(after.st_mode)),
            "owner": int(after.st_uid),
            "group": int(after.st_gid),
            "links": int(after.st_nlink),
            "size": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "ctime_ns": int(after.st_ctime_ns),
        }, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(
            b"dynamic-landmark-canonical-key-file-identity-v1\0" + material
        ).hexdigest()
    finally:
        os.close(descriptor)


def _with_key_identity(authorization: object, key_path: Path) -> object:
    private_key = getattr(authorization, "private_key", None)
    identity = _key_identity_sha256(
        key_path, private_key,
    )
    return _AuthorizedView(
        authorization=authorization,
        key_file_identity_sha256=identity,
    )


def _authorization_factories(args: argparse.Namespace) -> tuple[Callable[[], object], Callable[[], object]]:
    ravdess_key = _canonical(args.ravdess_key)
    mayo_key = _canonical(args.mayo_key)
    ravdess_captured: list[object] = []
    mayo_captured: list[object] = []

    def ravdess() -> object:
        try:
            authorized = authorize_committed_ravdess_semantic23(
                args.ravdess_data_root, id_key_path=ravdess_key,
            )
            ravdess_captured.append(
                _authorization_privacy_snapshot(authorized, "trials")
            )
            return _with_key_identity(authorized, ravdess_key)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("RAVDESS live authorization failed") from None

    def mayo() -> object:
        try:
            authorized = authorize_committed_mayo_ssl_generation(
                args.mayo_data_root,
                args.mayo_existing_export_root,
                mayo_key,
                args.mayo_cache_root,
                args.mayo_exposure_manifest,
            )
            mayo_captured.append(
                _authorization_privacy_snapshot(authorized, "recordings")
            )
            return _with_key_identity(authorized, mayo_key)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("Mayo live authorization failed") from None

    setattr(ravdess, "captured_authorizations", ravdess_captured)
    setattr(mayo, "captured_authorizations", mayo_captured)
    return ravdess, mayo


def _add_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ravdess-data-root", type=Path, required=True)
    parser.add_argument("--ravdess-key", type=Path, required=True)
    parser.add_argument("--mayo-data-root", type=Path, required=True)
    parser.add_argument("--mayo-existing-export-root", type=Path, required=True)
    parser.add_argument("--mayo-cache-root", type=Path, required=True)
    parser.add_argument("--mayo-exposure-manifest", type=Path, required=True)
    parser.add_argument("--mayo-key", type=Path, required=True)


class _PathRedactingArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, "bridge command arguments are invalid\n")


def _parser() -> argparse.ArgumentParser:
    parser = _PathRedactingArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize-mayo-key")
    initialize.add_argument("--key-path", type=Path, required=True)

    inventory = subparsers.add_parser("inventory")
    _add_authorization_arguments(inventory)

    build = subparsers.add_parser("build-bundles")
    _add_authorization_arguments(build)
    build.add_argument("--output-root", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-stage")
    _add_authorization_arguments(freeze)
    freeze.add_argument("--bridge-root", type=Path, required=True)
    freeze.add_argument("--mode", choices=("smoke", "formal"), required=True)
    freeze.add_argument("--run-id")

    verify = subparsers.add_parser("verify-determinism")
    _add_authorization_arguments(verify)
    verify.add_argument("--bridge-root", type=Path, required=True)
    verify.add_argument("--run-root", type=Path)
    return parser


def _run_root(mode: str, run_id: str | None) -> Path:
    base = Path(os.path.abspath(os.path.expanduser(os.fspath(PRETRAINING_ROOT))))
    if mode == "smoke":
        if type(run_id) is not str or _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError("smoke freeze requires one safe explicit run ID")
        return base / "smoke" / run_id
    if mode == "formal":
        if run_id is not None:
            raise ValueError("formal freeze has one canonical namespace and no run ID")
        return base / "formal"
    raise ValueError("bridge run mode is unsupported")


def _mode_for_committed_run_root(value: Path) -> str:
    observed = _canonical(value)
    formal = _canonical(PRETRAINING_ROOT / "formal")
    if observed == formal:
        return "formal"
    smoke = _canonical(PRETRAINING_ROOT / "smoke")
    if observed.parent == smoke and _SAFE_RUN_ID.fullmatch(observed.name):
        return "smoke"
    raise ValueError("run-root verification is confined to a canonical mode namespace")


@dataclass(frozen=True)
class _PrivacyForbidden:
    tokens: tuple[bytes, ...] = dataclass_field(repr=False)


@dataclass(frozen=True)
class _CapturedMayoCommandResult:
    json_line: str = dataclass_field(repr=False)


def _add_token(tokens: set[bytes], value: bytes) -> None:
    if type(value) is bytes and len(value) >= 4:
        tokens.add(value)


def _add_binary_variants(tokens: set[bytes], value: bytes) -> None:
    if not value:
        return
    _add_token(tokens, value)
    _add_token(tokens, value.hex().encode("ascii"))
    _add_token(tokens, value.hex().upper().encode("ascii"))
    for encoded in (
        base64.b64encode(value),
        base64.urlsafe_b64encode(value),
    ):
        _add_token(tokens, encoded)
        _add_token(tokens, encoded.rstrip(b"="))


def _add_text_variants(tokens: set[bytes], value: str) -> None:
    if type(value) is not str or not value:
        return
    for normalized in dict.fromkeys((
        value,
        unicodedata.normalize("NFC", value),
        unicodedata.normalize("NFD", value),
    )):
        raw = normalized.encode("utf-8")
        _add_binary_variants(tokens, raw)
        for encoding in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
            _add_token(tokens, normalized.encode(encoding))


def _add_name_variants(tokens: set[bytes], value: str) -> None:
    if type(value) is not str or not value:
        return
    for normalized in dict.fromkeys((
        value,
        unicodedata.normalize("NFC", value),
        unicodedata.normalize("NFD", value),
    )):
        _add_token(tokens, normalized.encode("utf-8"))


def _add_sha256_variants(tokens: set[bytes], value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("live SHA-256 privacy fact is malformed")
    raw = bytes.fromhex(value)
    _add_token(tokens, value.lower().encode("ascii"))
    _add_token(tokens, value.upper().encode("ascii"))
    _add_token(tokens, raw)
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        _add_token(tokens, encoded)
        _add_token(tokens, encoded.rstrip(b"="))


def _root_text_representations(path: Path) -> tuple[str, ...]:
    original = os.fspath(path)
    absolute = os.path.abspath(os.path.expanduser(original))
    try:
        resolved = os.fspath(Path(absolute).resolve(strict=True))
    except (OSError, RuntimeError):
        resolved = os.fspath(Path(absolute).resolve(strict=False))
    values = (
        original,
        absolute,
        resolved,
        Path(absolute).name,
        os.path.relpath(absolute, os.fspath(ROOT)),
        os.path.relpath(absolute, os.getcwd()),
    )
    return tuple(value for value in dict.fromkeys(values) if value)


def _authorizations(value: object) -> tuple[object, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _add_authorization_tokens(
    tokens: set[bytes],
    authorization: object,
    collection_name: str,
) -> None:
    key = getattr(authorization, "private_key", None)
    if type(key) is bytes and len(key) == 32:
        _add_binary_variants(tokens, key)
        _add_binary_variants(tokens, hashlib.sha256(key).digest())
        _add_token(tokens, repr(key).encode("ascii"))
        _add_token(tokens, json.dumps(list(key), separators=(",", ":")).encode("ascii"))
    if isinstance(authorization, _AuthorizationPrivacySnapshot):
        for cache_sha256 in authorization.cache_sha256s:
            _add_sha256_variants(tokens, cache_sha256)
        for observed in authorization.sensitive_values:
            _add_name_variants(tokens, observed)
        return
    for item in tuple(getattr(authorization, collection_name, ())):
        cache_sha256 = getattr(item, "cache_sha256", None)
        if type(cache_sha256) is str and re.fullmatch(r"[0-9a-fA-F]{64}", cache_sha256):
            _add_sha256_variants(tokens, cache_sha256)
        for field in (
            "source_sha256", "raw_filename", "filename", "session_name",
            "patient_id", "patient_identity", "subject_id",
        ):
            observed = getattr(item, field, None)
            if type(observed) is str:
                _add_name_variants(tokens, observed)


def _add_mayo_inventory_tokens(tokens: set[bytes], inventory: object) -> None:
    for collection in (
        "video_instances", "long_unique_videos", "duplicate_videos",
        "short_videos", "arkit_trajectories",
    ):
        for asset in tuple(getattr(inventory, collection, ())):
            digest = getattr(asset, "source_sha256", None)
            if type(digest) is str and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                _add_sha256_variants(tokens, digest)
            for field in ("session_path", "path", "export_dir"):
                value = getattr(asset, field, None)
                if value is None:
                    continue
                path = Path(value)
                _add_name_variants(tokens, os.fspath(path))
                _add_name_variants(tokens, path.name)
                _add_name_variants(tokens, path.stem)
                if field == "session_path":
                    _add_name_variants(tokens, path.name)
    for collection in ("arkit_sessions", "metadata_only_sessions"):
        for value in tuple(getattr(inventory, collection, ())):
            path = Path(value)
            _add_name_variants(tokens, os.fspath(path))
            _add_name_variants(tokens, path.name)


def _build_live_forbidden_tokens(
    *,
    mayo_roots: tuple[Path, Path],
    ravdess_authorization: object,
    mayo_authorization: object,
    ravdess_inventory: object,
    mayo_inventory: object,
) -> _PrivacyForbidden:
    """Build the in-memory-only sensitive token set used by final scanning."""
    tokens: set[bytes] = set()
    for root in mayo_roots:
        for representation in _root_text_representations(Path(root)):
            _add_text_variants(tokens, representation)
    for authorization in _authorizations(ravdess_authorization):
        _add_authorization_tokens(tokens, authorization, "trials")
    for authorization in _authorizations(mayo_authorization):
        _add_authorization_tokens(tokens, authorization, "recordings")
    member_sha256 = getattr(ravdess_inventory, "member_sha256", None)
    if not isinstance(member_sha256, dict):
        raise ValueError("live privacy facts are incomplete")
    for name, digest in member_sha256.items():
        if type(name) is not str or type(digest) is not str:
            raise ValueError("live privacy facts are malformed")
        _add_name_variants(tokens, name)
        _add_name_variants(tokens, Path(name).name)
        _add_name_variants(tokens, Path(name).stem)
        _add_sha256_variants(tokens, digest)
    _add_mayo_inventory_tokens(tokens, mayo_inventory)
    for field in (
        "patient_id", "patient_identity", "session_id", "session_name",
        "subject_id", "source_sha256", "cache_sha256", "private_key",
        "key_bytes", "raw_filename",
    ):
        _add_token(tokens, field.encode("ascii"))
        _add_token(tokens, f'"{field}"'.encode("ascii"))
    total = sum(len(token) for token in tokens)
    if not tokens or len(tokens) > 40_000 or total > 4 * 1024 * 1024:
        raise ValueError("live privacy token set exceeds its fixed bound")
    return _PrivacyForbidden(tokens=tuple(sorted(tokens)))


class _ByteMatcher:
    """Bounded Aho-Corasick matcher with streaming state across chunks."""

    def __init__(self, patterns: Sequence[bytes]):
        self._goto: list[dict[int, int]] = [{}]
        self._fail: list[int] = [0]
        self._output: list[bool] = [False]
        for pattern in patterns:
            if type(pattern) is not bytes or not pattern:
                raise ValueError("privacy token is malformed")
            state = 0
            for byte in pattern:
                next_state = self._goto[state].get(byte)
                if next_state is None:
                    next_state = len(self._goto)
                    self._goto[state][byte] = next_state
                    self._goto.append({})
                    self._fail.append(0)
                    self._output.append(False)
                state = next_state
            self._output[state] = True
        queue: collections.deque[int] = collections.deque()
        for state in self._goto[0].values():
            queue.append(state)
        while queue:
            state = queue.popleft()
            for byte, target in self._goto[state].items():
                queue.append(target)
                fallback = self._fail[state]
                while fallback and byte not in self._goto[fallback]:
                    fallback = self._fail[fallback]
                self._fail[target] = self._goto[fallback].get(byte, 0)
                self._output[target] = (
                    self._output[target] or self._output[self._fail[target]]
                )

    def feed(self, payload: bytes, state: int = 0) -> tuple[int, bool]:
        for byte in payload:
            while state and byte not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(byte, 0)
            if self._output[state]:
                return state, True
        return state, False


@dataclass
class _ScanBudget:
    entries: int = 0
    total_bytes: int = 0
    archive_count: int = 0
    archive_entries: int = 0
    expanded_bytes: int = 0


@dataclass(frozen=True)
class _DirectoryChain:
    components: tuple[str, ...]
    descriptors: tuple[int, ...] = dataclass_field(repr=False)
    identities: tuple[tuple[int, int], ...]

    @property
    def root_fd(self) -> int:
        return self.descriptors[-1]


@dataclass(frozen=True)
class _ZipCentralPreflight:
    file_size: int
    archive_base: int
    eocd_offset: int
    central_start: int
    central_size: int
    member_count: int


@dataclass(frozen=True)
class _TreeScanCommitment:
    kind: str
    stat_identity: tuple[int, ...]
    names: tuple[str, ...] = ()
    sha256: str = ""


@dataclass
class _TreeScanLedger:
    commitments: dict[tuple[str, ...], _TreeScanCommitment] = dataclass_field(
        default_factory=dict,
    )

    def record(
        self,
        path: tuple[str, ...],
        commitment: _TreeScanCommitment,
    ) -> None:
        if path in self.commitments:
            raise ValueError("private tree scan ledger contains a duplicate path")
        self.commitments[path] = commitment


_MAX_SCAN_ENTRIES = 20_000
_MAX_SCAN_DEPTH = 12
_MAX_SCAN_FILE_BYTES = 256 * 1024 * 1024
_MAX_SCAN_TREE_BYTES = 1024 * 1024 * 1024
_MAX_ZIP_ENTRIES = 4096
_MAX_ZIP_ARCHIVES = 4096
_MAX_ZIP_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ZIP_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_NESTED_ZIP_BYTES = 32 * 1024 * 1024
_MAX_NESTED_ZIP_DEPTH = 2
_MAX_ZIP_EOCD_SEARCH_BYTES = 22 + 65_535 + 76
_MAX_ZIP_CENTRAL_BYTES = 32 * 1024 * 1024
_MAX_ZIP_CENTRAL_RECORD_BYTES = 132 * 1024
_ZIP_EOCD_SIGNATURES = (b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07")
_ZIP_ORDINARY_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP64_EXTRA_ID = 0x0001


def _is_zip_handle(handle) -> bool:
    position = handle.tell()
    try:
        handle.seek(0)
        return bool(zipfile.is_zipfile(handle))
    finally:
        handle.seek(position)


def _directory_flags() -> int:
    return (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_chain_nofollow(value: Path) -> _DirectoryChain:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise ValueError("private verification root must be absolute")
    components = tuple(path.parts[1:])
    descriptors: list[int] = []
    identities: list[tuple[int, int]] = []
    try:
        root_fd = os.open(path.anchor, _directory_flags())
        descriptors.append(root_fd)
        root_stat = os.fstat(root_fd)
        identities.append((int(root_stat.st_dev), int(root_stat.st_ino)))
        for component in components:
            if (
                type(component) is not str
                or component in {"", ".", ".."}
                or Path(component).name != component
            ):
                raise ValueError("private verification path component is unsafe")
            parent_fd = descriptors[-1]
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            current = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(before.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError("private verification path chain is unsafe")
            identities.append((int(opened.st_dev), int(opened.st_ino)))
        return _DirectoryChain(
            components=components,
            descriptors=tuple(descriptors),
            identities=tuple(identities),
        )
    except Exception:
        _close_descriptor_sequence(tuple(descriptors))
        raise


def _assert_directory_chain(chain: _DirectoryChain) -> None:
    if (
        len(chain.descriptors) != len(chain.identities)
        or len(chain.descriptors) != len(chain.components) + 1
    ):
        raise ValueError("private verification path chain is malformed")
    for descriptor, identity in zip(chain.descriptors, chain.identities):
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode) or (
            int(current.st_dev), int(current.st_ino)
        ) != identity:
            raise ValueError("private verification path chain changed")
    for index, component in enumerate(chain.components, start=1):
        linked = os.stat(
            component,
            dir_fd=chain.descriptors[index - 1],
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(linked.st_mode) or (
            int(linked.st_dev), int(linked.st_ino)
        ) != chain.identities[index]:
            raise ValueError("private verification path chain changed")


def _close_descriptor_sequence(descriptors: Sequence[int]) -> None:
    closer = ExitStack()
    for descriptor in descriptors:
        closer.callback(os.close, descriptor)
    closer.__exit__(*sys.exc_info())


def _close_directory_chain(chain: _DirectoryChain) -> None:
    _close_descriptor_sequence(chain.descriptors)


def _close_directory_chains(chains: Sequence[_DirectoryChain]) -> None:
    closer = ExitStack()
    for chain in chains:
        closer.callback(_close_directory_chain, chain)
    closer.__exit__(*sys.exc_info())


def _zip_uint(payload: bytes, offset: int, width: int) -> int:
    return int.from_bytes(payload[offset:offset + width], "little")


def _read_zip_at(handle, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0:
        raise ValueError("private archive metadata is out of bounds")
    handle.seek(offset)
    payload = handle.read(size)
    if type(payload) is not bytes or len(payload) != size:
        raise ValueError("private archive metadata is truncated")
    return payload


def _reject_zip64_extra(extra: bytes) -> None:
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise ValueError("private archive extra metadata is malformed")
        field_id = _zip_uint(extra, cursor, 2)
        field_size = _zip_uint(extra, cursor + 2, 2)
        cursor += 4
        if field_size > len(extra) - cursor:
            raise ValueError("private archive extra metadata is malformed")
        if field_id == _ZIP64_EXTRA_ID:
            raise ValueError("ZIP64 private archives are unsupported")
        cursor += field_size


def _validate_local_extra(
    extra: bytes,
    *,
    compressed_size: int,
    uncompressed_size: int,
) -> None:
    """Allow NumPy's redundant bounded size extra, but no ZIP64 framing."""
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise ValueError("private archive extra metadata is malformed")
        field_id = _zip_uint(extra, cursor, 2)
        field_size = _zip_uint(extra, cursor + 2, 2)
        cursor += 4
        if field_size > len(extra) - cursor:
            raise ValueError("private archive extra metadata is malformed")
        field = extra[cursor:cursor + field_size]
        if field_id == _ZIP64_EXTRA_ID and (
            field_size != 16
            or _zip_uint(field, 0, 8) != uncompressed_size
            or _zip_uint(field, 8, 8) != compressed_size
        ):
            raise ValueError("ZIP64 private archive metadata is unsupported")
        cursor += field_size


def _preflight_zip_central_directory(
    handle,
    *,
    entry_limit: int,
) -> _ZipCentralPreflight:
    """Bound and cross-check an ordinary single-disk ZIP before ZipFile parses it."""
    if type(entry_limit) is not int or not 0 <= entry_limit <= _MAX_ZIP_ENTRIES:
        raise ValueError("private archive entry budget is invalid")
    original_position = handle.tell()
    try:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if type(file_size) is not int or file_size < 22:
            raise ValueError("private archive has no ordinary EOCD")
        tail_start = max(0, file_size - _MAX_ZIP_EOCD_SEARCH_BYTES)
        tail = _read_zip_at(handle, tail_start, file_size - tail_start)
        relative_eocd = tail.rfind(_ZIP_ORDINARY_EOCD_SIGNATURE)
        eocd_offset = -1
        eocd = b""
        while relative_eocd >= 0:
            candidate_offset = tail_start + relative_eocd
            if len(tail) - relative_eocd >= 22:
                candidate = tail[relative_eocd:relative_eocd + 22]
                comment_size = _zip_uint(candidate, 20, 2)
                if candidate_offset + 22 + comment_size == file_size:
                    eocd_offset = candidate_offset
                    eocd = candidate
                    break
            relative_eocd = tail.rfind(
                _ZIP_ORDINARY_EOCD_SIGNATURE, 0, relative_eocd,
            )
        if eocd_offset < 0:
            raise ValueError("private archive has no bounded ordinary EOCD")

        disk_number = _zip_uint(eocd, 4, 2)
        central_disk = _zip_uint(eocd, 6, 2)
        disk_members = _zip_uint(eocd, 8, 2)
        member_count = _zip_uint(eocd, 10, 2)
        central_size = _zip_uint(eocd, 12, 4)
        central_offset = _zip_uint(eocd, 16, 4)
        if (
            disk_number != 0
            or central_disk != 0
            or disk_members != member_count
        ):
            raise ValueError("multi-disk private archives are unsupported")
        if (
            disk_members == 0xFFFF
            or member_count == 0xFFFF
            or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF
        ):
            raise ValueError("ZIP64 private archives are unsupported")
        if member_count > entry_limit:
            raise ValueError("private archive has too many shared members")
        if central_size > _MAX_ZIP_CENTRAL_BYTES:
            raise ValueError("private archive central directory exceeds its bound")
        if eocd_offset >= 20 and _read_zip_at(
            handle, eocd_offset - 20, 4,
        ) == _ZIP64_LOCATOR_SIGNATURE:
            raise ValueError("ZIP64 private archives are unsupported")

        central_start = eocd_offset - central_size
        archive_base = central_start - central_offset
        if central_start < 0 or archive_base < 0 or archive_base > central_start:
            raise ValueError("private archive central directory is inconsistent")

        cursor = central_start
        actual_count = 0
        local_offsets: set[int] = set()
        while cursor < eocd_offset:
            if eocd_offset - cursor < 46:
                raise ValueError("private archive central directory is truncated")
            fixed = _read_zip_at(handle, cursor, 46)
            if fixed[:4] != _ZIP_CENTRAL_SIGNATURE:
                raise ValueError("private archive central directory is inconsistent")
            actual_count += 1
            if actual_count > member_count or actual_count > entry_limit:
                raise ValueError("private archive central member count is inconsistent")

            version_needed = _zip_uint(fixed, 6, 2)
            flags = _zip_uint(fixed, 8, 2)
            compression = _zip_uint(fixed, 10, 2)
            compressed_size = _zip_uint(fixed, 20, 4)
            uncompressed_size = _zip_uint(fixed, 24, 4)
            name_size = _zip_uint(fixed, 28, 2)
            extra_size = _zip_uint(fixed, 30, 2)
            comment_size = _zip_uint(fixed, 32, 2)
            start_disk = _zip_uint(fixed, 34, 2)
            local_offset = _zip_uint(fixed, 42, 4)
            record_size = 46 + name_size + extra_size + comment_size
            if (
                record_size > _MAX_ZIP_CENTRAL_RECORD_BYTES
                or cursor + record_size > eocd_offset
                or not 0 < name_size <= 1024
                or version_needed >= 45
                or flags & 0x1
                or compression not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or compressed_size == 0xFFFFFFFF
                or uncompressed_size == 0xFFFFFFFF
                or start_disk != 0
                or local_offset == 0xFFFFFFFF
                or local_offset in local_offsets
            ):
                raise ValueError("private archive central member is unsafe")
            local_offsets.add(local_offset)
            variable = _read_zip_at(handle, cursor + 46, record_size - 46)
            central_name = variable[:name_size]
            central_extra = variable[name_size:name_size + extra_size]
            if b"\x00" in central_name:
                raise ValueError("private archive central member name is unsafe")
            _reject_zip64_extra(central_extra)

            local_absolute = archive_base + local_offset
            if (
                local_absolute < archive_base
                or local_absolute + 30 > central_start
            ):
                raise ValueError("private archive local header is out of bounds")
            local = _read_zip_at(handle, local_absolute, 30)
            if local[:4] != _ZIP_LOCAL_SIGNATURE:
                raise ValueError("private archive local header is inconsistent")
            local_version = _zip_uint(local, 4, 2)
            local_flags = _zip_uint(local, 6, 2)
            local_compression = _zip_uint(local, 8, 2)
            local_compressed_size = _zip_uint(local, 18, 4)
            local_uncompressed_size = _zip_uint(local, 22, 4)
            local_name_size = _zip_uint(local, 26, 2)
            local_extra_size = _zip_uint(local, 28, 2)
            local_header_size = 30 + local_name_size + local_extra_size
            if (
                local_version >= 45
                or local_flags != flags
                or local_compression != compression
                or local_name_size != name_size
                or local_header_size > _MAX_ZIP_CENTRAL_RECORD_BYTES
                or local_absolute + local_header_size > central_start
            ):
                raise ValueError("private archive local header is unsafe")
            local_variable = _read_zip_at(
                handle, local_absolute + 30, local_header_size - 30,
            )
            local_name = local_variable[:local_name_size]
            local_extra = local_variable[local_name_size:]
            if local_name != central_name:
                raise ValueError("private archive member names are inconsistent")
            _validate_local_extra(
                local_extra,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
            )
            if not flags & 0x8 and (
                local_compressed_size != compressed_size
                or local_uncompressed_size != uncompressed_size
            ):
                raise ValueError("private archive member sizes are inconsistent")
            if flags & 0x8 and (
                local_compressed_size not in {0, compressed_size}
                or local_uncompressed_size not in {0, uncompressed_size}
            ):
                raise ValueError("private archive descriptor sizes are inconsistent")
            if local_absolute + local_header_size + compressed_size > central_start:
                raise ValueError("private archive member data is out of bounds")
            cursor += record_size

        if cursor != eocd_offset or actual_count != member_count:
            raise ValueError("private archive central member count is inconsistent")
        return _ZipCentralPreflight(
            file_size=file_size,
            archive_base=archive_base,
            eocd_offset=eocd_offset,
            central_start=central_start,
            central_size=central_size,
            member_count=member_count,
        )
    finally:
        handle.seek(original_position)


def _scan_zip(
    handle,
    matcher: _ByteMatcher,
    *,
    depth: int,
    budget: _ScanBudget,
) -> None:
    if depth > _MAX_NESTED_ZIP_DEPTH:
        raise ValueError("private archive nesting exceeds its bound")
    budget.archive_count += 1
    if budget.archive_count > _MAX_ZIP_ARCHIVES:
        raise ValueError("private archive count exceeds its shared bound")
    remaining_entries = _MAX_ZIP_ENTRIES - budget.archive_entries
    preflight = _preflight_zip_central_directory(
        handle,
        entry_limit=remaining_entries,
    )
    budget.archive_entries += preflight.member_count
    handle.seek(0)
    try:
        archive = zipfile.ZipFile(handle, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("private archive is invalid") from exc
    with archive:
        members = archive.infolist()
        if (
            len(members) != preflight.member_count
            or len(members) > _MAX_ZIP_ENTRIES
            or budget.archive_entries > _MAX_ZIP_ENTRIES
        ):
            raise ValueError("private archive has too many members")
        seen: set[str] = set()
        declared_total = 0
        for member in members:
            name = member.filename
            normalized = unicodedata.normalize("NFC", name).casefold()
            parts = name.split("/")
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(unix_mode)
            if (
                type(name) is not str
                or not name
                or len(name.encode("utf-8")) > 1024
                or "\x00" in name
                or "\\" in name
                or name.startswith("/")
                or re.match(r"^[A-Za-z]:", name)
                or any(part in {"", ".", ".."} for part in parts[:-1])
                or any(part == ".." for part in parts)
                or normalized in seen
                or member.flag_bits & 0x1
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError("private archive member is unsafe")
            seen.add(normalized)
            _, leaked_name = matcher.feed(name.encode("utf-8"))
            if leaked_name:
                raise ValueError("private archive member name contains sensitive data")
            if member.is_dir():
                raise ValueError("private archive directory members are unsupported")
            if (
                member.file_size < 0
                or member.file_size > _MAX_ZIP_MEMBER_BYTES
                or member.compress_size < 0
                or member.compress_size > _MAX_SCAN_FILE_BYTES
            ):
                raise ValueError("private archive member exceeds its bound")
            declared_total += member.file_size
            if declared_total > _MAX_ZIP_TOTAL_BYTES:
                raise ValueError("private archive expansion exceeds its bound")
            capture = (
                bytearray()
                if member.file_size <= _MAX_NESTED_ZIP_BYTES
                else None
            )
            archive_tail = bytearray() if capture is None else None
            actual = 0
            state = 0
            try:
                stream = archive.open(member, "r")
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ValueError("private archive member cannot be read safely") from exc
            with stream:
                while chunk := stream.read(1024 * 1024):
                    actual += len(chunk)
                    budget.expanded_bytes += len(chunk)
                    if actual > member.file_size or actual > _MAX_ZIP_MEMBER_BYTES:
                        raise ValueError("private archive member exceeded declared size")
                    if budget.expanded_bytes > _MAX_ZIP_TOTAL_BYTES:
                        raise ValueError(
                            "private archive shared expansion exceeds its bound"
                        )
                    state, leaked = matcher.feed(chunk, state)
                    if leaked:
                        raise ValueError("private archive member contains sensitive data")
                    if capture is not None:
                        capture.extend(chunk)
                    elif archive_tail is not None:
                        archive_tail.extend(chunk)
                        if len(archive_tail) > _MAX_ZIP_EOCD_SEARCH_BYTES:
                            del archive_tail[:-_MAX_ZIP_EOCD_SEARCH_BYTES]
            if actual != member.file_size:
                raise ValueError("private archive member size is inconsistent")
            if capture is not None:
                nested = io.BytesIO(bytes(capture))
                if _is_zip_handle(nested):
                    if depth >= _MAX_NESTED_ZIP_DEPTH:
                        raise ValueError("private archive nesting exceeds its bound")
                    _scan_zip(
                        nested,
                        matcher,
                        depth=depth + 1,
                        budget=budget,
                    )
            elif archive_tail is not None and any(
                signature in archive_tail for signature in _ZIP_EOCD_SIGNATURES
            ):
                raise ValueError(
                    "nested archive exceeds the bounded recursive scan size"
                )


def _integrity_stat(value) -> tuple[int, ...]:
    """Stat fields that bind storage identity/content without read-side atime."""
    birthtime_ns = getattr(value, "st_birthtime_ns", None)
    if birthtime_ns is None:
        birthtime = getattr(value, "st_birthtime", None)
        birthtime_ns = -1 if birthtime is None else round(float(birthtime) * 1e9)
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_rdev), int(value.st_size), int(value.st_mtime_ns),
        int(value.st_ctime_ns), int(getattr(value, "st_blksize", -1)),
        int(getattr(value, "st_blocks", -1)),
        int(getattr(value, "st_flags", -1)),
        int(getattr(value, "st_gen", -1)), int(birthtime_ns),
    )


def _scan_regular_fd(
    descriptor: int,
    size: int,
    matcher: _ByteMatcher,
    budget: _ScanBudget,
) -> str:
    if size < 0 or size > _MAX_SCAN_FILE_BYTES:
        raise ValueError("private file exceeds its scan bound")
    os.lseek(descriptor, 0, os.SEEK_SET)
    state = 0
    total = 0
    digest = hashlib.sha256()
    snapshot = bytearray()
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > size or total > _MAX_SCAN_FILE_BYTES:
            raise ValueError("private file changed during scanning")
        digest.update(chunk)
        state, leaked = matcher.feed(chunk, state)
        if leaked:
            raise ValueError("private file contains sensitive data")
        snapshot.extend(chunk)
    if total != size:
        raise ValueError("private file size changed during scanning")
    handle = io.BytesIO(bytes(snapshot))
    if _is_zip_handle(handle):
        _scan_zip(handle, matcher, depth=0, budget=budget)
    return digest.hexdigest()


def _hash_regular_fd(descriptor: int, expected_size: int) -> str:
    if expected_size < 0 or expected_size > _MAX_SCAN_FILE_BYTES:
        raise ValueError("private file exceeds its verification bound")
    os.lseek(descriptor, 0, os.SEEK_SET)
    total = 0
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > expected_size or total > _MAX_SCAN_FILE_BYTES:
            raise ValueError("private file changed during verification")
        digest.update(chunk)
    if total != expected_size:
        raise ValueError("private file size changed during verification")
    return digest.hexdigest()


def _scan_directory_fd(
    directory_fd: int,
    matcher: _ByteMatcher,
    budget: _ScanBudget,
    ledger: _TreeScanLedger,
    *,
    prefix: tuple[str, ...] = (),
    depth: int = 0,
) -> int:
    if depth > _MAX_SCAN_DEPTH:
        raise ValueError("private tree nesting exceeds its bound")
    directory_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise ValueError("private tree directory descriptor is unsafe")
    names = tuple(sorted(os.listdir(directory_fd)))
    ledger.record(
        prefix,
        _TreeScanCommitment(
            kind="directory",
            stat_identity=_integrity_stat(directory_before),
            names=names,
        ),
    )
    non_0600 = 0
    for name in names:
        if type(name) is not str or name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("private tree entry name is unsafe")
        relative = prefix + (name,)
        _, leaked_name = matcher.feed("/".join(relative).encode("utf-8"))
        if leaked_name:
            raise ValueError("private tree entry name contains sensitive data")
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        budget.entries += 1
        if budget.entries > _MAX_SCAN_ENTRIES:
            raise ValueError("private tree has too many entries")
        if stat.S_ISDIR(before.st_mode):
            if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o700:
                raise ValueError("private directory is not exact owner-only storage")
            child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if _integrity_stat(opened) != _integrity_stat(before):
                    raise ValueError("private directory changed while it was opened")
                non_0600 += _scan_directory_fd(
                    child_fd,
                    matcher,
                    budget,
                    ledger,
                    prefix=relative,
                    depth=depth + 1,
                )
                after = os.fstat(child_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    _integrity_stat(after) != _integrity_stat(before)
                    or _integrity_stat(current) != _integrity_stat(before)
                ):
                    raise ValueError("private directory changed during scanning")
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("private tree contains unsafe storage")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(descriptor)
            if (
                _integrity_stat(opened) != _integrity_stat(before)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
            ):
                raise ValueError("private file changed while it was opened")
            if opened.st_uid != os.geteuid() or stat.S_IMODE(opened.st_mode) != 0o600:
                non_0600 += 1
            budget.total_bytes += opened.st_size
            if budget.total_bytes > _MAX_SCAN_TREE_BYTES:
                raise ValueError("private tree exceeds its aggregate scan bound")
            sha256 = _scan_regular_fd(
                descriptor, opened.st_size, matcher, budget,
            )
            after = os.fstat(descriptor)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                _integrity_stat(opened) != _integrity_stat(after)
                or _integrity_stat(after) != _integrity_stat(current)
            ):
                raise ValueError("private file changed during scanning")
            ledger.record(
                relative,
                _TreeScanCommitment(
                    kind="file",
                    stat_identity=_integrity_stat(after),
                    sha256=sha256,
                ),
            )
        finally:
            os.close(descriptor)
    directory_after = os.fstat(directory_fd)
    final_names = tuple(sorted(os.listdir(directory_fd)))
    if (
        _integrity_stat(directory_after) != _integrity_stat(directory_before)
        or final_names != names
    ):
        raise ValueError("private directory changed during scanning")
    return non_0600


def _verify_tree_scan_ledger(
    root_fd: int,
    ledger: _TreeScanLedger,
) -> None:
    if not isinstance(ledger, _TreeScanLedger):
        raise ValueError("private tree scan ledger is malformed")
    visited: set[tuple[str, ...]] = set()

    def verify_directory(directory_fd: int, prefix: tuple[str, ...]) -> None:
        commitment = ledger.commitments.get(prefix)
        if commitment is None or commitment.kind != "directory":
            raise ValueError("private tree scan ledger directory is missing")
        if _integrity_stat(os.fstat(directory_fd)) != commitment.stat_identity:
            raise ValueError("private directory changed after scanning")
        names = tuple(sorted(os.listdir(directory_fd)))
        if names != commitment.names:
            raise ValueError("private directory entries changed after scanning")
        visited.add(prefix)
        for name in names:
            child_path = prefix + (name,)
            child = ledger.commitments.get(child_path)
            if child is None:
                raise ValueError("private tree scan ledger entry is missing")
            linked = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _integrity_stat(linked) != child.stat_identity:
                raise ValueError("private tree entry changed after scanning")
            if child.kind == "directory":
                child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
                try:
                    if _integrity_stat(os.fstat(child_fd)) != child.stat_identity:
                        raise ValueError("private directory changed while reopening")
                    verify_directory(child_fd, child_path)
                    if _integrity_stat(os.fstat(child_fd)) != child.stat_identity:
                        raise ValueError("private directory changed during verification")
                    current = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False,
                    )
                    if _integrity_stat(current) != child.stat_identity:
                        raise ValueError("private directory link changed during verification")
                finally:
                    os.close(child_fd)
                continue
            if child.kind != "file" or len(child.sha256) != 64:
                raise ValueError("private tree scan ledger file is malformed")
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or _integrity_stat(opened) != child.stat_identity
                ):
                    raise ValueError("private file changed while reopening")
                observed_sha256 = _hash_regular_fd(descriptor, opened.st_size)
                after = os.fstat(descriptor)
                current = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False,
                )
                if (
                    observed_sha256 != child.sha256
                    or _integrity_stat(after) != child.stat_identity
                    or _integrity_stat(current) != child.stat_identity
                ):
                    raise ValueError("private file changed after scanning")
            finally:
                os.close(descriptor)
            visited.add(child_path)
        if (
            tuple(sorted(os.listdir(directory_fd))) != commitment.names
            or _integrity_stat(os.fstat(directory_fd)) != commitment.stat_identity
        ):
            raise ValueError("private directory changed during verification")

    verify_directory(root_fd, ())
    if visited != set(ledger.commitments):
        raise ValueError("private tree scan ledger has unreachable entries")


def _scan_private_trees(
    roots: Sequence[Path],
    *,
    forbidden: _PrivacyForbidden,
) -> tuple[bool, bool, int]:
    if not isinstance(forbidden, _PrivacyForbidden):
        raise ValueError("live privacy authorization is required")
    matcher = _ByteMatcher(forbidden.tokens)
    non_0600 = 0
    budget = _ScanBudget()
    chains: list[_DirectoryChain] = []
    ledgers: list[_TreeScanLedger] = []
    try:
        for root in roots:
            chain = _open_directory_chain_nofollow(Path(root))
            chains.append(chain)
            _assert_directory_chain(chain)
            root_stat = os.fstat(chain.root_fd)
            if (
                not stat.S_ISDIR(root_stat.st_mode)
                or root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700
            ):
                raise ValueError("private verification root is unsafe")
        for chain in chains:
            _assert_directory_chain(chain)
            ledger = _TreeScanLedger()
            non_0600 += _scan_directory_fd(
                chain.root_fd, matcher, budget, ledger,
            )
            _assert_directory_chain(chain)
            ledgers.append(ledger)
        for chain, ledger in zip(chains, ledgers):
            _assert_directory_chain(chain)
            _verify_tree_scan_ledger(chain.root_fd, ledger)
            _assert_directory_chain(chain)
        for chain, ledger in reversed(tuple(zip(chains, ledgers))):
            _assert_directory_chain(chain)
            _verify_tree_scan_ledger(chain.root_fd, ledger)
            _assert_directory_chain(chain)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise ValueError("private artifact verification failed") from None
    finally:
        _close_directory_chains(chains)
    return non_0600 == 0, True, non_0600


def _json_line(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _live_privacy_inventories(args: argparse.Namespace) -> tuple[object, None]:
    """Audit RAVDESS live; Mayo inventory comes from its held v4 authorizer."""
    try:
        ravdess = audit_ravdess_inventory(args.ravdess_data_root)
        return ravdess, None
    except (OSError, RuntimeError, ValueError):
        raise ValueError("live privacy inventory authorization failed") from None


def _authorized_mayo_privacy_inventory(
    authorizations: Sequence[object],
) -> object:
    if type(authorizations) not in {tuple, list} or not authorizations:
        raise ValueError("authorized Mayo privacy inventory is missing")
    inventories = tuple(
        getattr(item, "privacy_inventory", None)
        for item in authorizations
    )
    if any(item is None for item in inventories):
        raise ValueError("authorized Mayo privacy inventory is incomplete")
    first = inventories[0]
    if any(item != first for item in inventories[1:]):
        raise ValueError("authorized Mayo privacy inventory changed")
    return first


def _mayo_cli_root_forbidden(args: argparse.Namespace) -> _PrivacyForbidden:
    tokens: set[bytes] = set()
    for root in (args.mayo_data_root, args.mayo_existing_export_root):
        for representation in _root_text_representations(Path(root)):
            _add_text_variants(tokens, representation)
            for normalized in dict.fromkeys((
                representation,
                unicodedata.normalize("NFC", representation),
                unicodedata.normalize("NFD", representation),
            )):
                escaped = json.dumps(
                    normalized,
                    ensure_ascii=True,
                    allow_nan=False,
                )[1:-1]
                _add_token(tokens, escaped.encode("ascii"))
    total = sum(len(token) for token in tokens)
    if not tokens or len(tokens) > 1024 or total > 1024 * 1024:
        raise ValueError("private Mayo command failed")
    return _PrivacyForbidden(tokens=tuple(sorted(tokens)))


def _capture_contains_forbidden(
    capture: object,
    forbidden: _PrivacyForbidden,
) -> bool:
    descriptor = capture.fileno()
    size = int(os.fstat(descriptor).st_size)
    if size < 0 or size > _MAX_MAYO_CLI_CAPTURE_BYTES:
        raise ValueError("captured command output exceeds its fixed bound")
    capture.seek(0)
    matcher = _ByteMatcher(forbidden.tokens)
    state = 0
    remaining = size
    while remaining:
        chunk = capture.read(min(1024 * 1024, remaining))
        if type(chunk) is not bytes or not chunk:
            raise ValueError("captured command output is truncated")
        remaining -= len(chunk)
        state, matched = matcher.feed(chunk, state)
        if matched:
            return True
    if capture.read(1) != b"":
        raise ValueError("captured command output changed during scanning")
    return False


def _record_cleanup_error(
    errors: list[BaseException],
    action: Callable[[], object],
) -> None:
    try:
        action()
    except BaseException as exc:  # fail closed while attempting later cleanup
        errors.append(exc)


def _restore_captured_descriptor(source: int, destination: int) -> None:
    try:
        os.dup2(source, destination)
    except BaseException as first:
        try:
            os.dup2(source, destination)
        except BaseException as second:
            raise RuntimeError("captured descriptor restoration failed") from second
        raise first


def _close_captured_descriptor(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except BaseException as first:
        try:
            os.fstat(descriptor)
        except BaseException:
            raise first
        try:
            os.close(descriptor)
        except BaseException as second:
            raise RuntimeError("captured descriptor close failed") from second
        raise first


def _close_captured_resource(resource: object) -> None:
    try:
        resource.close()
    except BaseException as first:
        if bool(getattr(resource, "closed", False)):
            raise first
        try:
            resource.close()
        except BaseException as second:
            raise RuntimeError("captured resource close failed") from second
        raise first


@dataclass
class _BoundedCaptureDrain:
    sink: object = dataclass_field(repr=False)
    read_descriptor: int = dataclass_field(repr=False)
    stop: threading.Event = dataclass_field(
        default_factory=threading.Event, repr=False,
    )
    thread: threading.Thread | None = dataclass_field(default=None, repr=False)
    started: bool = False
    overflowed: bool = False
    stored_bytes: int = 0
    errors: list[BaseException] = dataclass_field(default_factory=list, repr=False)


def _write_bounded_capture_prefix(
    drain: _BoundedCaptureDrain,
    payload: bytes,
) -> None:
    remaining = _MAX_MAYO_CLI_CAPTURE_BYTES - drain.stored_bytes
    if len(payload) > remaining:
        drain.overflowed = True
    if remaining <= 0:
        return
    view = memoryview(payload)[:remaining]
    while view:
        written = drain.sink.write(view)
        if type(written) is not int or written <= 0 or written > len(view):
            raise OSError("bounded capture sink write failed")
        drain.stored_bytes += written
        view = view[written:]


def _drain_bounded_capture(drain: _BoundedCaptureDrain) -> None:
    sink_failed = False
    try:
        os.set_blocking(drain.read_descriptor, False)
        while True:
            try:
                payload = os.read(drain.read_descriptor, 64 * 1024)
            except BlockingIOError:
                if drain.stop.is_set():
                    break
                drain.stop.wait(0.01)
                continue
            except InterruptedError:
                continue
            if not payload:
                break
            if not sink_failed:
                try:
                    _write_bounded_capture_prefix(drain, payload)
                except BaseException as exc:
                    drain.errors.append(exc)
                    sink_failed = True
    except BaseException as exc:
        drain.errors.append(exc)
    finally:
        try:
            _close_captured_descriptor(drain.read_descriptor)
        except BaseException as exc:
            drain.errors.append(exc)


def _start_bounded_capture_drain(
    sink: object,
    read_descriptor: int,
) -> _BoundedCaptureDrain:
    drain = _BoundedCaptureDrain(
        sink=sink,
        read_descriptor=read_descriptor,
    )
    try:
        drain.thread = threading.Thread(
            target=_drain_bounded_capture,
            args=(drain,),
            name="mayo-cli-output-drain",
            daemon=True,
        )
        drain.thread.start()
    except BaseException as primary:
        try:
            _close_captured_descriptor(read_descriptor)
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    drain.started = True
    return drain


def _finish_bounded_capture_drain(
    drain: _BoundedCaptureDrain,
    errors: list[BaseException],
) -> None:
    drain.stop.set()
    if drain.started and drain.thread is not None:
        try:
            drain.thread.join(timeout=5.0)
            if drain.thread.is_alive():
                raise RuntimeError("bounded capture drain did not terminate")
        except BaseException as exc:
            errors.append(exc)
    else:
        _record_cleanup_error(
            errors,
            lambda: _close_captured_descriptor(drain.read_descriptor),
        )
    errors.extend(drain.errors)


def _run_mayo_cli_captured(
    args: argparse.Namespace,
    operation: Callable[[], object],
) -> _CapturedMayoCommandResult:
    """Run one Mayo-consuming command with Python and FD output quarantined."""
    forbidden: _PrivacyForbidden | None = None
    stdout_capture = None
    stderr_capture = None
    stdout_pipe_write: int | None = None
    stderr_pipe_write: int | None = None
    stdout_drain: _BoundedCaptureDrain | None = None
    stderr_drain: _BoundedCaptureDrain | None = None
    saved_stdout: int | None = None
    saved_stderr: int | None = None
    text_stdout_descriptor: int | None = None
    text_stderr_descriptor: int | None = None
    text_stdout = None
    text_stderr = None
    stdout_redirected = False
    stderr_redirected = False
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    result: object = None
    result_json_line: str | None = None
    leaked = False
    try:
        try:
            stdout_capture = tempfile.TemporaryFile(mode="w+b")
            stderr_capture = tempfile.TemporaryFile(mode="w+b")
            stdout_pipe_read, stdout_pipe_write = os.pipe()
            stdout_drain = _start_bounded_capture_drain(
                stdout_capture, stdout_pipe_read,
            )
            stderr_pipe_read, stderr_pipe_write = os.pipe()
            stderr_drain = _start_bounded_capture_drain(
                stderr_capture, stderr_pipe_read,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            saved_stdout = os.dup(1)
            saved_stderr = os.dup(2)
            os.dup2(stdout_pipe_write, 1)
            stdout_redirected = True
            os.dup2(stderr_pipe_write, 2)
            stderr_redirected = True
            text_stdout_descriptor = os.dup(stdout_pipe_write)
            text_stdout = os.fdopen(
                text_stdout_descriptor,
                "w",
                encoding="utf-8",
                errors="backslashreplace",
                buffering=1,
            )
            text_stdout_descriptor = None
            text_stderr_descriptor = os.dup(stderr_pipe_write)
            text_stderr = os.fdopen(
                text_stderr_descriptor,
                "w",
                encoding="utf-8",
                errors="backslashreplace",
                buffering=1,
            )
            text_stderr_descriptor = None
            with redirect_stdout(text_stdout), redirect_stderr(text_stderr):
                forbidden = _mayo_cli_root_forbidden(args)
                result = operation()
                result_payload = json.dumps(
                    result,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
                if len(result_payload) > _MAX_MAYO_CLI_CAPTURE_BYTES:
                    raise ValueError("captured command result exceeds its fixed bound")
                result_matcher = _ByteMatcher(forbidden.tokens)
                _state, result_leaked = result_matcher.feed(result_payload)
                leaked = leaked or result_leaked
                result_json_line = result_payload.decode("ascii")
        except BaseException as exc:
            primary = exc
    finally:
        for stream in (text_stderr, text_stdout):
            if stream is not None:
                _record_cleanup_error(cleanup_errors, stream.flush)
        if stderr_redirected:
            if saved_stderr is None:
                cleanup_errors.append(RuntimeError("stderr restoration is unavailable"))
            else:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda: _restore_captured_descriptor(saved_stderr, 2),
                )
        if stdout_redirected:
            if saved_stdout is None:
                cleanup_errors.append(RuntimeError("stdout restoration is unavailable"))
            else:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda: _restore_captured_descriptor(saved_stdout, 1),
                )
        for stream in (text_stderr, text_stdout):
            if stream is not None:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda stream=stream: _close_captured_resource(stream),
                )

        for descriptor in (text_stderr_descriptor, text_stdout_descriptor):
            if descriptor is not None:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda descriptor=descriptor: _close_captured_descriptor(
                        descriptor
                    ),
                )
        for descriptor in (stderr_pipe_write, stdout_pipe_write):
            if descriptor is not None:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda descriptor=descriptor: _close_captured_descriptor(
                        descriptor
                    ),
                )

        for drain in (stdout_drain, stderr_drain):
            if drain is not None:
                _finish_bounded_capture_drain(drain, cleanup_errors)

        for capture in (stdout_capture, stderr_capture):
            if capture is not None:
                _record_cleanup_error(cleanup_errors, capture.flush)

        for capture in (stdout_capture, stderr_capture):
            if capture is not None and forbidden is not None:
                try:
                    leaked = (
                        _capture_contains_forbidden(capture, forbidden) or leaked
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)

        if any(
            drain is not None and drain.overflowed
            for drain in (stdout_drain, stderr_drain)
        ):
            cleanup_errors.append(
                ValueError("captured command output exceeds its fixed bound")
            )

        for descriptor in (saved_stderr, saved_stdout):
            if descriptor is not None:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda descriptor=descriptor: _close_captured_descriptor(
                        descriptor
                    ),
                )
        for capture in (stderr_capture, stdout_capture):
            if capture is not None:
                _record_cleanup_error(
                    cleanup_errors,
                    lambda capture=capture: _close_captured_resource(capture),
                )

    if (
        primary is not None
        or cleanup_errors
        or leaked
        or type(result_json_line) is not str
    ):
        raise ValueError("private Mayo command failed") from None
    return _CapturedMayoCommandResult(json_line=result_json_line)


def _run_mayo_cli_operation(args: argparse.Namespace) -> object:
    ravdess_authorizer, mayo_authorizer = _authorization_factories(args)
    producer = _producer_sha256()
    if args.command == "inventory":
        ravdess = ravdess_authorizer()
        mayo = mayo_authorizer()
        return {
            "ravdess_trials": int(getattr(ravdess, "trial_count")),
            "ravdess_actors": int(getattr(ravdess, "actor_count")),
            "mayo_recordings": int(getattr(mayo, "recording_count")),
            "mayo_source_units": len({
                getattr(item, "recording_id") for item in getattr(mayo, "recordings")
            }),
        }

    canonical_bridge = PRETRAINING_ROOT / "bridge"
    if args.command == "build-bundles":
        output = _require_exact_path(
            args.output_root, canonical_bridge, "bridge output root",
        )
        stages = build_bridge_bundles(
            output,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer,
        )
        return {
            "bundle_count": 2,
            "mayo_samples": int(stages["mayo"]["sample_count"]),
            "ravdess_samples": int(stages["ravdess"]["sample_count"]),
        }

    bridge = _require_exact_path(
        args.bridge_root, canonical_bridge, "bridge root",
    )
    if args.command == "freeze-stage":
        run_root = _run_root(args.mode, args.run_id)
        return freeze_bridge_stage(
            run_root,
            bridge,
            mode=args.mode,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer,
        )

    if args.command == "verify-determinism":
        roots = [bridge]
        run_root: Path | None = None
        mode: str | None = None
        if args.run_root is not None:
            run_root = _canonical(args.run_root)
            mode = _mode_for_committed_run_root(run_root)
            roots.append(run_root)
        inventory_before: tuple[object, object] | None = None
        scan_result: tuple[bool, bool, int] | None = None

        def capture_inventory_before() -> None:
            nonlocal inventory_before
            if inventory_before is not None:
                raise ValueError("live privacy inventory was captured more than once")
            inventory_before = _live_privacy_inventories(args)

        def finalize_under_destination_locks() -> None:
            nonlocal scan_result
            if inventory_before is None:
                raise ValueError("live privacy inventory facts are missing")
            inventory_after = _live_privacy_inventories(args)
            if inventory_before[0] != inventory_after[0]:
                raise ValueError("live privacy inventory changed during verification")
            ravdess_captured = tuple(getattr(
                ravdess_authorizer, "captured_authorizations", (),
            ))
            mayo_captured = tuple(getattr(
                mayo_authorizer, "captured_authorizations", (),
            ))
            if not ravdess_captured or not mayo_captured:
                raise ValueError("live privacy authorization facts are missing")
            if inventory_before[1] is None and inventory_after[1] is None:
                mayo_privacy_inventory = _authorized_mayo_privacy_inventory(
                    mayo_captured,
                )
            else:
                if inventory_before[1] != inventory_after[1]:
                    raise ValueError(
                        "live privacy inventory changed during verification"
                    )
                mayo_privacy_inventory = inventory_after[1]
            forbidden = _build_live_forbidden_tokens(
                mayo_roots=(args.mayo_data_root, args.mayo_existing_export_root),
                ravdess_authorization=ravdess_captured,
                mayo_authorization=mayo_captured,
                ravdess_inventory=inventory_before[0],
                mayo_inventory=mayo_privacy_inventory,
            )
            scan_result = _scan_private_trees(roots, forbidden=forbidden)

        if run_root is not None and mode is not None:
            result = verify_frozen_bridge_stage(
                run_root / "inputs",
                bridge,
                mode=mode,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer,
                before_authorization=capture_inventory_before,
                finalize_locked=finalize_under_destination_locks,
                include_generation_result=True,
            )
        else:
            result = verify_bridge_generation(
                bridge,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer,
                before_authorization=capture_inventory_before,
                finalize_locked=finalize_under_destination_locks,
            )
        if scan_result is None:
            raise ValueError("locked privacy scan result is missing")
        modes_ok, privacy_ok, non_0600 = scan_result
        result["modes_ok"] = modes_ok
        result["privacy_ok"] = privacy_ok
        result["non_0600_private_file_count"] = non_0600
        if set(result) != {
            "bundle_count", "bundle_total_bytes", "deterministic", "modes_ok",
            "non_0600_private_file_count", "privacy_ok", "size_ok",
        } or not all(bool(result[name]) for name in (
            "deterministic", "modes_ok", "privacy_ok", "size_ok",
        )):
            raise ValueError("determinism verification did not satisfy all claims")
        return result
    raise RuntimeError("unreachable bridge CLI command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "initialize-mayo-key":
        requested_lexical = Path(os.path.abspath(os.path.expanduser(
            os.fspath(args.key_path)
        )))
        expected_lexical = Path(os.path.abspath(os.path.expanduser(
            os.fspath(CANONICAL_MAYO_KEY)
        )))
        if requested_lexical != expected_lexical:
            raise ValueError("Mayo key path is not the canonical private location")
        created = initialize_owner_only_key(expected_lexical)
        _json_line({"created": created, "key_bytes": 32, "mode": "0600"})
        return 0

    captured = _run_mayo_cli_captured(
        args, lambda: _run_mayo_cli_operation(args),
    )
    print(captured.json_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "_parser"]
