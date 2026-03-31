#!/usr/bin/env python3
"""Render Music Floor from one seed prompt plus nearby style-token alternatives."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

DEMO_DEFAULT_SEED_PROMPT = "minimal techno"
DEFAULT_TOKEN_BANK_PATH = REPO_ROOT / "token_embeddings" / "llm_large_style_token_banks.npy"
DEFAULT_NEIGHBOR_SKIP = 0


def _prepend_path(path: Path, package_name: str) -> bool:
    if not (path / package_name).is_dir():
        return False
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return True


def _ensure_projection_mapping_on_path() -> None:
    candidates = [
        REPO_ROOT.parent / "projector-mapping",
        Path.home() / "git" / "projector-mapping",
    ]
    for candidate in candidates:
        if _prepend_path(candidate, "projection_mapping"):
            return
    raise SystemExit(
        "Could not locate the sibling 'projector-mapping' repository. "
        "Expected it at ../projector-mapping or ~/git/projector-mapping."
    )


def _ensure_pose_stream_on_path() -> None:
    pose_candidates = [
        REPO_ROOT.parent / "3d_pose_estimation_optitrack" / "src",
        Path.home() / "git" / "3d_pose_estimation_optitrack" / "src",
    ]
    if not any(_prepend_path(candidate, "pose_stream") for candidate in pose_candidates):
        raise SystemExit(
            "Could not locate the sibling '3d_pose_estimation_optitrack' repo. "
            "Expected it at ../3d_pose_estimation_optitrack or ~/git/3d_pose_estimation_optitrack."
        )

    lunar_candidates = [
        REPO_ROOT.parent / "lunar_tools",
        Path.home() / "git" / "lunar_tools",
        REPO_ROOT.parent / "lunar_tools_refact",
        Path.home() / "git" / "lunar_tools_refact",
    ]
    if not any(_prepend_path(candidate, "lunar_tools") for candidate in lunar_candidates):
        raise SystemExit(
            "Could not locate the sibling 'lunar_tools' repo. "
            "Expected it at ../lunar_tools or ~/git/lunar_tools."
        )


_ensure_projection_mapping_on_path()

# Keep projector rendering on SDL display, but never let Pygame compete for the
# system audio device. Magenta playback already owns audio via sounddevice.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from projection_mapping import CanvasProjectorRouter
from projection_mapping.apply_correction import ProjectionAlignment
from projection_mapping.multi import load_room_layout

Size = Tuple[int, int]

WORLD_MIN = -3.5
WORLD_MAX = 3.5
GRID_COLUMNS = 6
GRID_ROWS = 6
STYLE_ROW_INDEX = 0
DEFAULT_ROW_INDEX = STYLE_ROW_INDEX
STYLE_TOKEN_MIN = 0
STYLE_TOKEN_MAX = 1023
DEFAULT_MAGENTA_SERVER_URL = "http://graciosa:8013"
DEFAULT_PLAYER_CONTROL_URL = "http://127.0.0.1:8014"
DEFAULT_POSE_PROTOCOL = "lucid"
DEFAULT_LUCID_POSE_SERVICE = "pose3d-yolo26-pose-yolo26l"
DEFAULT_LUCID_POSE_PORT = 8048
DEFAULT_ZMQ_POSE_PORT = 5570
DEFAULT_DECODE_WINDOW = 2
DEFAULT_CROSSFADE_FRAMES = 1
DEFAULT_TEMPERATURE = 1.1
DEFAULT_TOPK = 40
DEFAULT_GUIDANCE_WEIGHT = 5.0

BLACK = (0, 0, 0)
TEXT_MAIN = (235, 235, 235)
TEXT_MUTED = (160, 160, 160)
TEXT_ACCENT = (205, 205, 205)
TEXT_ERROR = (232, 120, 120)

FLOOR_COLUMN_FILL = (18, 18, 18)
FLOOR_COLUMN_BORDER = (42, 42, 42)
FLOOR_ACTIVE_FILL = (18, 48, 24)
FLOOR_ACTIVE_BORDER = (62, 138, 70)
FLOOR_SEPARATOR = (28, 28, 28)
FLOOR_CIRCLE_FILL = (132, 132, 132)
FLOOR_CIRCLE_BORDER = (188, 188, 188)
FLOOR_SELECTED_FILL = (58, 196, 84)
FLOOR_SELECTED_BORDER = (208, 255, 216)
FLOOR_MARKER = (255, 255, 255)
FLOOR_TOKEN_TEXT = (12, 12, 12)

FRONT_PANEL = (20, 20, 20)
FRONT_PANEL_BORDER = (80, 80, 80)
FRONT_PANEL_ACTIVE = (20, 56, 26)
FRONT_PANEL_ACTIVE_BORDER = (84, 190, 100)
FRONT_BUTTON_FILL = (34, 34, 34)
FRONT_BUTTON_BORDER = (110, 110, 110)
FRONT_BUTTON_TEXT = (236, 236, 236)
FRONT_BUTTON_HINT = (170, 170, 170)
PROMPT_EDIT_OVERLAY = (0, 0, 0, 176)
PROMPT_EDIT_PANEL = (22, 22, 22)
PROMPT_EDIT_PANEL_BORDER = (96, 96, 96)
PROMPT_EDIT_PANEL_ACTIVE = (28, 56, 32)
PROMPT_EDIT_PANEL_ACTIVE_BORDER = (84, 190, 100)
EMULATION_BG = (8, 8, 8)
EMULATION_MARGIN = 24
EMULATION_GAP = 24


@dataclass
class PersonFloorSample:
    person_id: int
    x: float
    z: float
    column_index: int
    row_index: int
    confidence: float


@dataclass
class GridState:
    selected_rows: List[int] = field(
        default_factory=lambda: [DEFAULT_ROW_INDEX for _ in range(GRID_COLUMNS)]
    )
    grid_tokens: List[List[int]] = field(
        default_factory=lambda: [
            [0 for _ in range(GRID_ROWS)] for _ in range(GRID_COLUMNS)
        ]
    )
    row_labels: List[str] = field(
        default_factory=lambda: [f"Row {index + 1}" for index in range(GRID_ROWS)]
    )
    row_prompts: List[str] = field(
        default_factory=lambda: ["" for _ in range(GRID_ROWS)]
    )
    row_style_tokens: List[List[int]] = field(
        default_factory=lambda: [
            [0 for _ in range(GRID_COLUMNS)] for _ in range(GRID_ROWS)
        ]
    )
    current_style_tokens: List[int] = field(
        default_factory=lambda: [0 for _ in range(GRID_COLUMNS)]
    )
    active_columns: set[int] = field(default_factory=set)
    people: List[PersonFloorSample] = field(default_factory=list)
    column_drivers: Dict[int, PersonFloorSample] = field(default_factory=dict)
    people_count: int = 0
    frame_index: int | None = None
    timestamp_iso: str | None = None
    pose_status: str = "static rows"
    prompt_bank_name: str = DEMO_DEFAULT_SEED_PROMPT
    magenta_server_url: str = DEFAULT_MAGENTA_SERVER_URL
    magenta_status: str = "magenta idle"
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    prompt_edit_active: bool = False
    prompt_edit_text: str = ""
    prompt_edit_status: str = "Press P to edit the seed prompt."


@dataclass(frozen=True)
class EmulationLayout:
    window_size: Size
    floor_rect: pygame.Rect
    front_rect: pygame.Rect | None


class ZmqPoseStreamSource:
    def __init__(
        self,
        endpoint: str,
        port: int,
        timeout: float,
        poll_interval: float,
    ) -> None:
        _ensure_pose_stream_on_path()
        from pose_stream import PoseStreamClient  # type: ignore

        self.endpoint = endpoint
        self.port = port
        self.label = f"pose zmq {endpoint}:{port}"
        self.poll_interval = max(0.001, poll_interval)
        self._client = PoseStreamClient(
            endpoint=endpoint,
            port=port,
            timeout=timeout,
        )

    def poll_latest(self):
        latest = None
        while True:
            message = self._client.receive(
                poll_interval=self.poll_interval,
                wait=False,
            )
            if message is None:
                return latest
            latest = message

    def wait_for_latest(self, max_wait_seconds: float):
        deadline = time.monotonic() + max(0.0, max_wait_seconds)
        latest = self.poll_latest()
        while latest is None and time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            latest = self.poll_latest()
        return latest

    def close(self) -> None:
        self._client.close()


class LucidPoseSource:
    def __init__(
        self,
        endpoint: str,
        port: int,
        timeout: float,
        poll_interval: float,
        service_name: str,
    ) -> None:
        self.endpoint = endpoint
        self.port = port
        self.service_name = (service_name or "").strip()
        if not self.service_name:
            raise ValueError("Lucid pose service name is required.")
        self.timeout = max(0.001, timeout)
        self.poll_interval = max(0.001, poll_interval)
        self.base_url = self._build_base_url(endpoint, port)
        self.label = f"pose lucid {self.service_name} @ {self.base_url}"
        self._last_sequence: int | None = None
        self._next_poll_at = 0.0

        info = self._request_json("/info")
        detected_service = str(info.get("service") or "").strip()
        if detected_service and detected_service != self.service_name:
            raise RuntimeError(
                f"Expected Lucid pose service {self.service_name!r}, "
                f"but {self.base_url} reports {detected_service!r}."
            )

    @staticmethod
    def _build_base_url(endpoint: str, port: int) -> str:
        host = (endpoint or "").strip()
        if not host:
            raise ValueError("Pose endpoint is required.")
        if "://" in host:
            parsed = urllib_parse.urlparse(host)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("Lucid pose endpoint must be a host or http(s) URL.")
            base_url = host.rstrip("/")
            if parsed.port is None and port:
                netloc = f"{parsed.hostname}:{int(port)}"
                if parsed.username:
                    auth = parsed.username
                    if parsed.password:
                        auth = f"{auth}:{parsed.password}"
                    netloc = f"{auth}@{netloc}"
                base_url = urllib_parse.urlunparse(
                    (
                        parsed.scheme,
                        netloc,
                        parsed.path or "",
                        parsed.params or "",
                        parsed.query or "",
                        parsed.fragment or "",
                    )
                ).rstrip("/")
            return base_url
        return f"http://{host}:{int(port)}"

    def _request_json(self, path: str) -> Dict[str, Any]:
        request = urllib_request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET {path} failed: {exc.code} {details}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"GET {path} failed: {exc.reason}") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GET {path} returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"GET {path} returned {type(data).__name__}, expected object.")
        return data

    def poll_latest(self):
        now = time.monotonic()
        if now < self._next_poll_at:
            return None
        self._next_poll_at = now + self.poll_interval

        payload = self._request_json("/poses/latest")
        message = payload.get("message")
        sequence = payload.get("sequence")
        try:
            sequence_value = int(sequence) if sequence is not None else None
        except (TypeError, ValueError):
            sequence_value = None

        if (
            sequence_value is not None
            and self._last_sequence is not None
            and sequence_value <= self._last_sequence
        ):
            return None
        if sequence_value is not None:
            self._last_sequence = sequence_value

        if not isinstance(message, dict):
            return None
        return message

    def wait_for_latest(self, max_wait_seconds: float):
        deadline = time.monotonic() + max(0.0, max_wait_seconds)
        latest = self.poll_latest()
        while latest is None and time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            latest = self.poll_latest()
        return latest

    def close(self) -> None:
        return None


PoseSource = ZmqPoseStreamSource | LucidPoseSource


def _resolve_token_bank_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def load_style_token_banks(path_text: str) -> tuple[Path, np.ndarray]:
    path = _resolve_token_bank_path(path_text)
    if not path.is_file():
        raise ValueError(f"Style token bank file not found: {path}")
    token_banks = np.load(path)
    if token_banks.ndim != 3:
        raise ValueError(
            f"Expected style token banks with 3 dims, got shape {token_banks.shape}."
        )
    if token_banks.shape[0] != GRID_COLUMNS or token_banks.shape[1] != STYLE_TOKEN_MAX + 1:
        raise ValueError(
            "Style token banks must have shape "
            f"({GRID_COLUMNS}, {STYLE_TOKEN_MAX + 1}, embedding_dim); "
            f"got {token_banks.shape}."
        )
    if token_banks.shape[2] <= 0:
        raise ValueError("Style token banks must have a non-zero embedding dimension.")
    return path, np.asarray(token_banks, dtype=np.float32)


def _normalize_token_banks(token_banks: np.ndarray) -> np.ndarray:
    normalized = np.asarray(token_banks, dtype=np.float32).copy()
    norms = np.linalg.norm(normalized, axis=2, keepdims=True)
    np.divide(normalized, np.maximum(norms, 1e-12), out=normalized)
    return normalized


def build_neighbor_row_style_tokens(
    orig_tokens: List[int],
    normalized_token_banks: np.ndarray,
    *,
    skip_nearest: int,
) -> List[List[int]]:
    if len(orig_tokens) != GRID_COLUMNS:
        raise ValueError(
            f"Expected {GRID_COLUMNS} original tokens, got {len(orig_tokens)}."
        )
    if normalized_token_banks.shape[0] != GRID_COLUMNS:
        raise ValueError(
            f"Expected {GRID_COLUMNS} style-token banks, got {normalized_token_banks.shape[0]}."
        )
    token_count = int(normalized_token_banks.shape[1])
    if skip_nearest < 0:
        raise ValueError("--skip-nearest must be >= 0.")
    if skip_nearest > token_count - GRID_ROWS:
        raise ValueError(
            f"--skip-nearest={skip_nearest} is too large for {token_count} tokens."
        )

    column_tokens: List[List[int]] = []
    for column_index, orig_token in enumerate(orig_tokens):
        orig_token_id = int(orig_token)
        if orig_token_id < 0 or orig_token_id >= token_count:
            raise ValueError(
                f"Original token {orig_token_id} at column {column_index + 1} "
                f"is outside 0..{token_count - 1}."
            )
        bank = normalized_token_banks[column_index]
        scores = bank @ bank[orig_token_id]
        ranked = np.argsort(scores)[::-1]
        alternates: List[int] = []
        skipped = 0
        for candidate in ranked:
            token_id = int(candidate)
            if token_id == orig_token_id:
                continue
            if skipped < skip_nearest:
                skipped += 1
                continue
            alternates.append(token_id)
            if len(alternates) == GRID_ROWS - 1:
                break
        if len(alternates) != GRID_ROWS - 1:
            raise ValueError(
                f"Could not find {GRID_ROWS - 1} alternates for column {column_index + 1}."
            )
        column_tokens.append([orig_token_id, *alternates])

    return [
        [column_tokens[column_index][row_index] for column_index in range(GRID_COLUMNS)]
        for row_index in range(GRID_ROWS)
    ]


class MagentaController:
    def __init__(
        self,
        server_url: str,
        player_control_url: str | None,
        seed_prompt: str,
        token_bank_path: str,
        skip_nearest: int,
        decode_window: int,
        context_frames: int | None,
        crossfade_frames: int,
        temperature: float,
        topk: int,
        guidance_weight: float,
    ) -> None:
        self.skip_nearest = int(skip_nearest)
        self.server_url = self._normalize_server_url(server_url)
        self.player_control_url = self._normalize_optional_url(player_control_url)
        self._use_player_control = False
        self.token_bank_path, self.token_banks = load_style_token_banks(token_bank_path)
        self.normalized_token_banks = _normalize_token_banks(self.token_banks)
        self.stream_info = self._fetch_stream_info()
        self.generation_kwargs = self._build_generation_kwargs(
            decode_window=decode_window,
            context_frames=context_frames,
            crossfade_frames=crossfade_frames,
            temperature=temperature,
            topk=topk,
            guidance_weight=guidance_weight,
        )
        self.seed_prompt = ""
        self.seed_tokens: List[int] = []
        self.row_labels: List[str] = []
        self.row_prompts: List[str] = []
        self.row_style_tokens: List[List[int]] = []
        self.reseed_prompt(seed_prompt)
        self.current_style_tokens: List[int] = list(self.row_style_tokens[DEFAULT_ROW_INDEX])
        self._latest_status = "server control idle"
        self._refresh_player_control_mode()
        self._apply_control(self.current_style_tokens)

    @staticmethod
    def _normalize_server_url(server_url: str) -> str:
        url = (server_url or "").strip().rstrip("/")
        if not url:
            raise ValueError("Magenta server URL is required.")
        parsed = urllib_parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Magenta server URL must start with http:// or https://")
        return url

    @classmethod
    def _normalize_optional_url(cls, server_url: str | None) -> str | None:
        if server_url is None:
            return None
        text = (server_url or "").strip()
        if not text:
            return None
        return cls._normalize_server_url(text)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        base_url: str | None = None,
    ):
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib_request.Request(
            f"{base_url or self.server_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                payload = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {details}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return json.loads(payload)

    def _refresh_player_control_mode(self) -> None:
        if self.player_control_url is None or self._use_player_control:
            return
        try:
            response = self._request_json(
                "GET",
                "/state",
                base_url=self.player_control_url,
            )
        except Exception:
            return
        if not isinstance(response, dict):
            return
        response_server_url = str(response.get("server_url") or "").strip()
        if response_server_url:
            self.server_url = self._normalize_server_url(response_server_url)
        response_generation_kwargs = response.get("generation_kwargs")
        if isinstance(response_generation_kwargs, dict):
            self.generation_kwargs = dict(response_generation_kwargs)
        status_text = str(response.get("status_text") or "").strip()
        if status_text:
            self._latest_status = status_text
        self._use_player_control = True

    def _fetch_stream_info(self) -> Dict[str, Any]:
        response = self._request_json("GET", "/stream_info")
        if not isinstance(response, dict):
            raise RuntimeError("Invalid /stream_info response: expected object.")
        return response

    def _build_generation_kwargs(
        self,
        *,
        decode_window: int,
        context_frames: int | None,
        crossfade_frames: int,
        temperature: float,
        topk: int,
        guidance_weight: float,
    ) -> Dict[str, Any]:
        stream_info = self.stream_info
        frame_length_samples = int(stream_info.get("frame_length_samples", 1920))
        max_decode_frames = max(1, int(stream_info.get("chunk_length_frames", 50)))
        default_context_frames = int(stream_info.get("context_length_frames", 250))
        min_context_frames = int(stream_info.get("min_context_length_frames", 25))
        max_context_frames = int(stream_info.get("max_context_length_frames", default_context_frames))
        max_crossfade_frames = int(stream_info.get("max_crossfade_length_frames", 5))
        decode_value = max(1, min(int(round(decode_window)), max_decode_frames))
        if context_frames is None:
            context_value = default_context_frames
        else:
            context_value = int(round(context_frames))
        context_value = max(min_context_frames, min(max_context_frames, context_value))
        crossfade_value = max(0, min(int(round(crossfade_frames)), max_crossfade_frames))
        return {
            "max_decode_frames": decode_value,
            "context_length_frames": context_value,
            "crossfade_samples": crossfade_value * frame_length_samples,
            "temperature": max(0.0, min(4.0, float(temperature))),
            "topk": max(0, min(1024, int(round(topk)))),
            "guidance_weight": max(0.0, min(10.0, float(guidance_weight))),
            "seed": None,
        }

    @staticmethod
    def _normalize_prompt_text(prompt: str) -> str:
        normalized_prompt = " ".join((prompt or "").strip().split())
        if not normalized_prompt:
            raise ValueError("Seed prompt is required.")
        return normalized_prompt

    def _build_seed_prompt_state(
        self,
        prompt: str,
    ) -> tuple[str, List[int], List[str], List[str], List[List[int]]]:
        normalized_prompt = self._normalize_prompt_text(prompt)
        seed_tokens = self._tokenize_style_text(normalized_prompt)
        row_labels = ["Orig"] + [f"Alt {index}" for index in range(1, GRID_ROWS)]
        row_prompts = [normalized_prompt] + [
            f"Per-column cosine neighbor rank {self.skip_nearest + index}"
            for index in range(1, GRID_ROWS)
        ]
        row_style_tokens = build_neighbor_row_style_tokens(
            seed_tokens,
            self.normalized_token_banks,
            skip_nearest=self.skip_nearest,
        )
        return normalized_prompt, seed_tokens, row_labels, row_prompts, row_style_tokens

    def _tokenize_style_text(self, prompt: str) -> List[int]:
        stream_info = self.stream_info
        token_count = int(stream_info.get("style_token_count", GRID_COLUMNS))
        if token_count != GRID_COLUMNS:
            raise RuntimeError(
                f"Magenta server expects {token_count} style tokens, "
                f"but Music Floor uses {GRID_COLUMNS} columns."
            )
        payload = prompt.encode("utf-8")
        response = self._request_json(
            "POST",
            "/style_tokens",
            data=payload,
            content_type="text/plain; charset=utf-8",
        )
        if not isinstance(response, list) or len(response) != GRID_COLUMNS:
            raise RuntimeError(
                f"Invalid /style_tokens response: expected {GRID_COLUMNS} tokens."
            )
        return [int(token) for token in response]

    def reseed_prompt(self, prompt: str) -> None:
        (
            normalized_prompt,
            seed_tokens,
            row_labels,
            row_prompts,
            row_style_tokens,
        ) = self._build_seed_prompt_state(prompt)
        self.seed_prompt = normalized_prompt
        self.seed_tokens = seed_tokens
        self.row_labels = row_labels
        self.row_prompts = row_prompts
        self.row_style_tokens = row_style_tokens

    def _consume_control_response(
        self,
        response,
        fallback_tokens: List[int],
        *,
        status_prefix: str,
    ) -> None:
        if isinstance(response, dict) and isinstance(response.get("style_tokens"), list):
            self.current_style_tokens = [int(token) for token in response["style_tokens"]]
        else:
            self.current_style_tokens = [int(token) for token in fallback_tokens]
        if isinstance(response, dict) and isinstance(response.get("generation_kwargs"), dict):
            self.generation_kwargs = dict(response["generation_kwargs"])
        if isinstance(response, dict):
            status_text = str(response.get("status_text") or "").strip()
            if status_text:
                self._latest_status = status_text
                return
        session_is_active = bool(response.get("session_is_active")) if isinstance(response, dict) else False
        playback_state = str(response.get("playback_state") or "unknown") if isinstance(response, dict) else "unknown"
        session_text = "session active" if session_is_active else "no player session"
        self._latest_status = (
            f"{status_prefix} {session_text} | playback={playback_state.lower()} @ {self.server_url}"
        )

    def _apply_control(self, style_tokens: List[int]) -> None:
        self._refresh_player_control_mode()
        payload = {
            "style_tokens": [int(token) for token in style_tokens],
            "style_text": self.seed_prompt,
            "style_source": "manual",
        }
        if self._use_player_control and self.player_control_url is not None:
            response = self._request_json(
                "POST",
                "/update_style_tokens",
                data=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                base_url=self.player_control_url,
            )
            status_prefix = "player control"
        else:
            response = self._request_json(
                "POST",
                "/control",
                data=json.dumps(
                    {
                        "style_tokens": [int(token) for token in style_tokens],
                        "generation_kwargs": dict(self.generation_kwargs),
                    }
                ).encode("utf-8"),
                content_type="application/json",
            )
            status_prefix = "server control"
        self._consume_control_response(
            response,
            [int(token) for token in style_tokens],
            status_prefix=status_prefix,
        )

    def poll(self) -> None:
        return None

    def update_style_tokens(self, style_tokens: List[int]) -> List[int]:
        tokens = [int(token) for token in style_tokens]
        self._apply_control(tokens)
        return list(self.current_style_tokens)

    def reset_context(self, style_tokens: List[int] | None = None) -> List[int]:
        tokens = (
            [int(token) for token in style_tokens]
            if style_tokens is not None
            else list(self.current_style_tokens)
        )
        self._refresh_player_control_mode()
        if self._use_player_control and self.player_control_url is not None:
            response = self._request_json(
                "POST",
                "/reset_session",
                data=json.dumps(
                    {
                        "style_tokens": tokens,
                        "style_text": self.seed_prompt,
                        "style_source": "manual",
                    }
                ).encode("utf-8"),
                content_type="application/json",
                base_url=self.player_control_url,
            )
            status_prefix = "player reset"
        else:
            self._apply_control(tokens)
            response = self._request_json(
                "POST",
                "/reset_context",
                data=b"{}",
                content_type="application/json",
            )
            status_prefix = "context reset"
        self._consume_control_response(
            response,
            tokens,
            status_prefix=status_prefix,
        )
        return list(self.current_style_tokens)

    def status_text(self) -> str:
        return self._latest_status

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Floor demo with one seed prompt plus nearby style-token alternatives."
    )
    parser.add_argument(
        "--config",
        default="cork",
        help="Room config describing canvases/projectors.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second while displaying.",
    )
    parser.add_argument(
        "--floor-canvas",
        default="floor",
        help="Canvas name for the logical floor surface.",
    )
    parser.add_argument(
        "--front-canvas",
        default="front",
        help="Canvas name for the front surface.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        help="Optional directory where raw canvas previews are saved as PNGs.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render and save previews, then exit without opening projector windows.",
    )
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="Run in a single local Pygame window and emulate floor tracking from mouse clicks.",
    )
    parser.add_argument(
        "--pose-endpoint",
        type=str,
        default="iki",
        help="Pose host/IP. For Lucid this is the service host; for ZMQ it is the stream host.",
    )
    parser.add_argument(
        "--pose-protocol",
        choices=("lucid", "zmq"),
        default=DEFAULT_POSE_PROTOCOL,
        help="Pose transport to use. Defaults to Lucid HTTP.",
    )
    parser.add_argument(
        "--pose-service",
        type=str,
        default=DEFAULT_LUCID_POSE_SERVICE,
        help="Lucid pose service name when --pose-protocol=lucid.",
    )
    parser.add_argument(
        "--pose-port",
        type=int,
        default=None,
        help="Pose TCP port. Defaults to 8048 for Lucid or 5570 for ZMQ.",
    )
    parser.add_argument(
        "--pose-timeout",
        type=float,
        default=1.0,
        help="Pose stream socket timeout in seconds.",
    )
    parser.add_argument(
        "--pose-poll-interval",
        type=float,
        default=0.02,
        help="Pose polling sleep interval in seconds.",
    )
    parser.add_argument(
        "--pose-bootstrap-seconds",
        type=float,
        default=0.0,
        help="Optional initial wait time for a pose frame before rendering.",
    )
    parser.add_argument(
        "--seed-prompt",
        type=str,
        default=DEMO_DEFAULT_SEED_PROMPT,
        help="Single seed prompt used to derive the original six style tokens.",
    )
    parser.add_argument(
        "--music-prompt",
        type=str,
        default=None,
        help="Legacy alias for --seed-prompt.",
    )
    parser.add_argument(
        "--token-bank-path",
        type=str,
        default=str(DEFAULT_TOKEN_BANK_PATH),
        help="Path to the six style-token embedding banks (.npy).",
    )
    parser.add_argument(
        "--skip-nearest",
        type=int,
        default=DEFAULT_NEIGHBOR_SKIP,
        help="Skip this many nearest neighbors before filling Alt rows.",
    )
    parser.add_argument(
        "--decode-window",
        type=int,
        default=DEFAULT_DECODE_WINDOW,
        help="Emit frames per step, aligned with the Gradio client default.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Context window in codec frames. Defaults to the server/Gradio default.",
    )
    parser.add_argument(
        "--crossfade-frames",
        type=int,
        default=DEFAULT_CROSSFADE_FRAMES,
        help="Crossfade size in codec frames, aligned with the Gradio client default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature, aligned with the Gradio client default.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Top-k sampling cutoff, aligned with the Gradio client default.",
    )
    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=DEFAULT_GUIDANCE_WEIGHT,
        help="Guidance weight, aligned with the Gradio client default.",
    )
    parser.add_argument(
        "--player-control-url",
        type=str,
        default=DEFAULT_PLAYER_CONTROL_URL,
        help="Local music_floor_player control URL. Empty string disables it and falls back to direct server control.",
    )
    parser.add_argument(
        "--magenta-server-url",
        type=str,
        default=DEFAULT_MAGENTA_SERVER_URL,
        help="URL of the Magenta realtime server.",
    )
    parser.add_argument(
        "--debug-person",
        action="append",
        default=[],
        help="Inject a synthetic floor position as x,z for testing. Repeatable.",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        default=True,
        help="Enable optimized per-channel projector transforms.",
    )
    return parser.parse_args()


def _grid_centers(count: int) -> List[float]:
    span = WORLD_MAX - WORLD_MIN
    step = span / count
    return [WORLD_MIN + step * (index + 0.5) for index in range(count)]


def _grid_step(count: int) -> float:
    return (WORLD_MAX - WORLD_MIN) / count


def _world_to_pixel(size: Size, x_world: float, y_world: float) -> Tuple[int, int]:
    width, height = size
    span = WORLD_MAX - WORLD_MIN
    x_norm = (x_world - WORLD_MIN) / span
    y_norm = (WORLD_MAX - y_world) / span
    x_px = int(round(x_norm * width))
    y_px = int(round(y_norm * height))
    return x_px, y_px


def _coord_to_index(value: float, count: int, *, invert: bool) -> int | None:
    if value < WORLD_MIN or value > WORLD_MAX:
        return None
    span = WORLD_MAX - WORLD_MIN
    normalized = (value - WORLD_MIN) / span
    if invert:
        normalized = 1.0 - normalized
    index = int(normalized * count)
    return max(0, min(count - 1, index))


def _column_index_from_x(x: float) -> int | None:
    return _coord_to_index(x, GRID_COLUMNS, invert=False)


def _row_index_from_z(z: float) -> int | None:
    return _coord_to_index(z, GRID_ROWS, invert=True)


def _row_label(row_index: int) -> str:
    return f"Row {row_index + 1}"


def _build_token_columns(
    row_style_tokens: List[List[int]],
) -> List[List[int]]:
    if len(row_style_tokens) != GRID_ROWS:
        raise SystemExit(
            f"Expected {GRID_ROWS} prompt rows, got {len(row_style_tokens)}."
        )

    columns = [[0 for _ in range(GRID_ROWS)] for _ in range(GRID_COLUMNS)]
    for row_index, row_tokens in enumerate(row_style_tokens):
        if len(row_tokens) != GRID_COLUMNS:
            raise SystemExit(
                f"Prompt row {row_index + 1} must have {GRID_COLUMNS} style tokens, "
                f"got {len(row_tokens)}."
            )
        for column_index, token in enumerate(row_tokens):
            columns[column_index][row_index] = int(token)

    return columns


def _selected_style_tokens(state: GridState) -> List[int]:
    return [
        state.grid_tokens[column_index][state.selected_rows[column_index]]
        for column_index in range(GRID_COLUMNS)
    ]


def _effective_canvas_sizes(config_path: str) -> Dict[str, Size]:
    layout = load_room_layout(config_path)
    sizes = {name: spec.size for name, spec in layout.canvases.items()}
    for spec in layout.projectors:
        try:
            alignment = ProjectionAlignment.load(
                spec.config_path,
                projector_id=spec.id,
            )
        except ValueError:
            continue
        if alignment.canvas_size is not None:
            sizes[spec.canvas] = alignment.canvas_size
    return sizes


def _pick_floor_canvas(canvas_sizes: Dict[str, Size], requested: str) -> str:
    if requested in canvas_sizes:
        return requested
    if requested == "floor" and "floor" in canvas_sizes:
        return "floor"
    candidates = [name for name in sorted(canvas_sizes) if "floor" in name.lower()]
    if candidates:
        return candidates[0]
    raise SystemExit(
        f"Could not resolve a floor canvas from {requested!r}; available={sorted(canvas_sizes)}"
    )


def _font(size: int) -> pygame.font.Font:
    return pygame.font.SysFont(None, max(12, size))


def _parse_debug_people(raw_values: Iterable[str]) -> List[PersonFloorSample]:
    samples: List[PersonFloorSample] = []
    for index, raw in enumerate(raw_values, start=1):
        text = raw.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            raise SystemExit(
                f"Invalid --debug-person value {raw!r}; expected x,z such as --debug-person=-2.1,0.3"
            )
        try:
            x = float(parts[0])
            z = float(parts[1])
        except ValueError as exc:
            raise SystemExit(f"Invalid --debug-person value {raw!r}: {exc}") from exc
        sample = _sample_from_floor_point(
            person_id=index,
            x=x,
            z=z,
            confidence=1.0,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _sample_from_floor_point(
    person_id: int,
    x: float,
    z: float,
    confidence: float,
) -> PersonFloorSample | None:
    column_index = _column_index_from_x(x)
    row_index = _row_index_from_z(z)
    if column_index is None or row_index is None:
        return None
    return PersonFloorSample(
        person_id=person_id,
        x=x,
        z=z,
        column_index=column_index,
        row_index=row_index,
        confidence=confidence,
    )


def _message_value(payload: Any, key: str, default=None):
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _message_items(payload: Any, key: str) -> List[Any]:
    value = _message_value(payload, key, [])
    return value if isinstance(value, list) else []


def _message_mapping(payload: Any, key: str) -> Dict[str, Any]:
    value = _message_value(payload, key, {})
    return value if isinstance(value, dict) else {}


def _extract_position_from_keypoint(keypoint) -> Tuple[float, float, float] | None:
    position = _message_value(keypoint, "position", None)
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError):
        return None


def _extract_floor_samples_from_message(message) -> List[PersonFloorSample]:
    samples: List[PersonFloorSample] = []
    for person in _message_items(message, "people"):
        ankle_positions: List[Tuple[float, float, float]] = []
        ankle_confidences: List[float] = []
        keypoints = _message_mapping(person, "keypoints")
        for keypoint_name in ("left_ankle", "right_ankle"):
            keypoint = keypoints.get(keypoint_name)
            if keypoint is None:
                continue
            position = _extract_position_from_keypoint(keypoint)
            if position is None:
                continue
            ankle_positions.append(position)
            confidence = _message_value(keypoint, "confidence", None)
            try:
                if confidence is not None:
                    ankle_confidences.append(float(confidence))
            except (TypeError, ValueError):
                pass

        if not ankle_positions:
            continue

        x = sum(position[0] for position in ankle_positions) / len(ankle_positions)
        z = sum(position[2] for position in ankle_positions) / len(ankle_positions)
        confidence = (
            sum(ankle_confidences) / len(ankle_confidences)
            if ankle_confidences
            else 0.0
        )

        sample = _sample_from_floor_point(
            person_id=int(_message_value(person, "id", 0)),
            x=z,
            z=x,
            confidence=confidence,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _apply_samples_to_state(
    state: GridState,
    samples: List[PersonFloorSample],
    *,
    pose_status: str,
    frame_index: int | None = None,
    timestamp_iso: str | None = None,
) -> List[int]:
    state.people = list(samples)
    state.people_count = len(samples)
    state.active_columns = {sample.column_index for sample in samples}
    state.pose_status = pose_status
    if frame_index is not None:
        state.frame_index = frame_index
    if timestamp_iso is not None:
        state.timestamp_iso = timestamp_iso

    best_by_column: Dict[int, PersonFloorSample] = {}
    for sample in sorted(samples, key=lambda item: (-item.confidence, item.person_id)):
        existing = best_by_column.get(sample.column_index)
        if existing is None or sample.confidence > existing.confidence:
            best_by_column[sample.column_index] = sample

    changed_columns: List[int] = []
    for column_index, sample in best_by_column.items():
        if state.selected_rows[column_index] != sample.row_index:
            state.selected_rows[column_index] = sample.row_index
            changed_columns.append(column_index)
    state.column_drivers = best_by_column
    return changed_columns


def _render_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: Tuple[int, int, int],
    x: int,
    y: int,
) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max(0, max_chars - 1)].rstrip()}…"


def _front_reset_button_rect(size: Size) -> pygame.Rect:
    width, height = size
    left_pad = int(round(width * 0.06))
    top_pad = int(round(height * 0.07))
    button_width = max(180, int(round(width * 0.22)))
    button_height = max(42, int(round(height * 0.075)))
    return pygame.Rect(
        width - left_pad - button_width,
        top_pad,
        button_width,
        button_height,
    )


def _front_prompt_button_rect(size: Size) -> pygame.Rect:
    reset_rect = _front_reset_button_rect(size)
    gap = max(18, int(round(size[0] * 0.015)))
    return pygame.Rect(
        reset_rect.left - gap - reset_rect.width,
        reset_rect.top,
        reset_rect.width,
        reset_rect.height,
    )


def _scale_rect_to_dest(
    rect: pygame.Rect,
    src_size: Size,
    dest_rect: pygame.Rect,
) -> pygame.Rect:
    src_width = max(1, int(src_size[0]))
    src_height = max(1, int(src_size[1]))
    return pygame.Rect(
        dest_rect.left + int(round(rect.left * dest_rect.width / src_width)),
        dest_rect.top + int(round(rect.top * dest_rect.height / src_height)),
        max(1, int(round(rect.width * dest_rect.width / src_width))),
        max(1, int(round(rect.height * dest_rect.height / src_height))),
    )


def _emulation_reset_button_hit(
    *,
    layout: EmulationLayout,
    front_size: Size,
    mouse_position: Tuple[int, int],
) -> bool:
    if layout.front_rect is None:
        return False
    if not layout.front_rect.collidepoint(mouse_position):
        return False
    button_rect = _front_reset_button_rect(front_size)
    scaled_button_rect = _scale_rect_to_dest(button_rect, front_size, layout.front_rect)
    return scaled_button_rect.collidepoint(mouse_position)


def _emulation_prompt_button_hit(
    *,
    layout: EmulationLayout,
    front_size: Size,
    mouse_position: Tuple[int, int],
) -> bool:
    if layout.front_rect is None:
        return False
    if not layout.front_rect.collidepoint(mouse_position):
        return False
    button_rect = _front_prompt_button_rect(front_size)
    scaled_button_rect = _scale_rect_to_dest(button_rect, front_size, layout.front_rect)
    return scaled_button_rect.collidepoint(mouse_position)


def _wrap_text(
    text: str,
    font: pygame.font.Font,
    max_width: int,
) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return [""]
    words = cleaned.split(" ")
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def create_floor_frame(size: Size, state: GridState) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill(BLACK)
    width, height = size

    x_centers = _grid_centers(GRID_COLUMNS)
    y_centers = list(reversed(_grid_centers(GRID_ROWS)))
    cell_w = width / GRID_COLUMNS
    cell_h = height / GRID_ROWS
    radius = max(8, int(round(min(cell_w, cell_h) * 0.24)))
    token_font = _font(max(12, int(round(radius * 0.78))))
    lane_width = max(24, int(round(cell_w * 0.64)))
    lane_top = int(round(cell_h * 0.35))
    lane_height = max(1, height - lane_top * 2)
    lane_border = max(2, int(round(radius * 0.08)))
    separator_width = max(2, int(round(cell_w * 0.03)))

    for column_index, x_world in enumerate(x_centers):
        x_px, _ = _world_to_pixel(size, x_world, 0.0)
        lane_rect = pygame.Rect(0, 0, lane_width, lane_height)
        lane_rect.center = (x_px, height // 2)
        is_active = column_index in state.active_columns
        lane_fill = FLOOR_ACTIVE_FILL if is_active else FLOOR_COLUMN_FILL
        lane_border_color = FLOOR_ACTIVE_BORDER if is_active else FLOOR_COLUMN_BORDER
        pygame.draw.rect(
            surface,
            lane_fill,
            lane_rect,
            border_radius=max(12, lane_width // 2),
        )
        pygame.draw.rect(
            surface,
            lane_border_color,
            lane_rect,
            width=lane_border,
            border_radius=max(12, lane_width // 2),
        )

    for column_index in range(GRID_COLUMNS - 1):
        boundary_world = (x_centers[column_index] + x_centers[column_index + 1]) / 2.0
        x_px, _ = _world_to_pixel(size, boundary_world, 0.0)
        pygame.draw.line(
            surface,
            FLOOR_SEPARATOR,
            (x_px, int(round(cell_h * 0.25))),
            (x_px, height - int(round(cell_h * 0.25))),
            separator_width,
        )

    for column_index, x_world in enumerate(x_centers):
        selected_row = state.selected_rows[column_index]
        for row_index, z_world in enumerate(y_centers):
            center = _world_to_pixel(size, x_world, z_world)
            is_selected = row_index == selected_row
            token = state.grid_tokens[column_index][row_index]
            fill = FLOOR_SELECTED_FILL if is_selected else FLOOR_CIRCLE_FILL
            border = FLOOR_SELECTED_BORDER if is_selected else FLOOR_CIRCLE_BORDER
            pygame.draw.circle(surface, fill, center, radius)
            pygame.draw.circle(
                surface,
                border,
                center,
                radius,
                width=max(2, int(round(radius * 0.08))),
            )
            token_surface = token_font.render(str(token), True, FLOOR_TOKEN_TEXT)
            token_rect = token_surface.get_rect(center=center)
            surface.blit(token_surface, token_rect)

    marker_radius = max(6, int(round(min(cell_w, cell_h) * 0.08)))
    for sample in state.people:
        marker_center = _world_to_pixel(size, sample.x, sample.z)
        pygame.draw.circle(surface, FLOOR_MARKER, marker_center, marker_radius, width=2)

    return surface


def create_front_frame(
    size: Size,
    state: GridState,
    *,
    show_reset_button: bool = False,
) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill(BLACK)
    width, height = size

    title_font = _font(max(28, height // 12))
    subtitle_font = _font(max(18, height // 24))
    row_font = _font(max(14, height // 36))
    panel_title_font = _font(max(18, height // 22))
    panel_value_font = _font(max(22, height // 15))
    panel_status_font = _font(max(14, height // 30))

    top_pad = int(round(height * 0.07))
    left_pad = int(round(width * 0.06))
    _render_text(surface, "Music Floor", title_font, TEXT_MAIN, left_pad, top_pad)
    _render_text(
        surface,
        "Row 0 keeps the seed prompt tokens. Lower rows swap in nearby token alternatives per style position.",
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + 10,
    )

    frame_label = (
        f"people={state.people_count} frame={state.frame_index}"
        if state.frame_index is not None
        else f"people={state.people_count}"
    )
    _render_text(
        surface,
        f"{state.pose_status}  |  {frame_label}",
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() + 26,
    )
    _render_text(
        surface,
        f"Seed prompt: {_truncate_text(state.prompt_bank_name, 72)}",
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 2 + 38,
    )
    _render_text(
        surface,
        state.prompt_edit_status,
        row_font,
        TEXT_ERROR if "failed" in state.prompt_edit_status.lower() else TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 3 + 44,
    )
    _render_text(
        surface,
        f"Live style tokens: {state.current_style_tokens}",
        subtitle_font,
        TEXT_ACCENT,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 4 + 50,
    )
    if state.generation_kwargs:
        crossfade_samples = int(state.generation_kwargs.get("crossfade_samples", 0))
        crossfade_frames = int(round(crossfade_samples / 1920.0)) if crossfade_samples else 0
        generation_label = (
            "Gen: "
            f"emit={int(state.generation_kwargs.get('max_decode_frames', 0))} "
            f"ctx={int(state.generation_kwargs.get('context_length_frames', 0))} "
            f"xfade={crossfade_frames} "
            f"temp={float(state.generation_kwargs.get('temperature', 0.0)):.2f} "
            f"topk={int(state.generation_kwargs.get('topk', 0))} "
            f"guide={float(state.generation_kwargs.get('guidance_weight', 0.0)):.2f}"
        )
        _render_text(
            surface,
            generation_label,
            row_font,
            TEXT_MUTED,
            left_pad,
            top_pad + title_font.get_height() + subtitle_font.get_height() * 5 + 58,
        )
    _render_text(
        surface,
        state.magenta_status,
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 6 + 70,
    )

    if show_reset_button:
        prompt_button_rect = _front_prompt_button_rect(size)
        pygame.draw.rect(surface, FRONT_BUTTON_FILL, prompt_button_rect, border_radius=16)
        pygame.draw.rect(
            surface,
            FRONT_BUTTON_BORDER,
            prompt_button_rect,
            width=3,
            border_radius=16,
        )
        prompt_label_surface = panel_title_font.render("Edit Seed Prompt", True, FRONT_BUTTON_TEXT)
        prompt_label_rect = prompt_label_surface.get_rect(
            center=(prompt_button_rect.centerx, prompt_button_rect.centery - 8)
        )
        surface.blit(prompt_label_surface, prompt_label_rect)
        prompt_hint_surface = panel_status_font.render("Press P", True, FRONT_BUTTON_HINT)
        prompt_hint_rect = prompt_hint_surface.get_rect(
            center=(prompt_button_rect.centerx, prompt_button_rect.centery + 16)
        )
        surface.blit(prompt_hint_surface, prompt_hint_rect)

        reset_button_rect = _front_reset_button_rect(size)
        pygame.draw.rect(surface, FRONT_BUTTON_FILL, reset_button_rect, border_radius=16)
        pygame.draw.rect(
            surface,
            FRONT_BUTTON_BORDER,
            reset_button_rect,
            width=3,
            border_radius=16,
        )
        reset_label_surface = panel_title_font.render("Reset Context", True, FRONT_BUTTON_TEXT)
        reset_label_rect = reset_label_surface.get_rect(
            center=(reset_button_rect.centerx, reset_button_rect.centery - 8)
        )
        surface.blit(reset_label_surface, reset_label_rect)
        reset_hint_surface = panel_status_font.render("Press R", True, FRONT_BUTTON_HINT)
        reset_hint_rect = reset_hint_surface.get_rect(
            center=(reset_button_rect.centerx, reset_button_rect.centery + 16)
        )
        surface.blit(reset_hint_surface, reset_hint_rect)

    legend_top = top_pad + title_font.get_height() + subtitle_font.get_height() * 6 + 82
    legend_col_gap = int(round(width * 0.03))
    legend_col_width = int(round((width - left_pad * 2 - legend_col_gap) / 2))
    legend_row_height = row_font.get_height() * 2 + 16
    for row_index in range(GRID_ROWS):
        col_index = row_index // 3
        row_in_col = row_index % 3
        any_selected = any(selected == row_index for selected in state.selected_rows)
        text_color = FLOOR_SELECTED_BORDER if any_selected else TEXT_MAIN
        detail_color = FLOOR_ACTIVE_BORDER if any_selected else TEXT_MUTED
        legend_x = left_pad + col_index * (legend_col_width + legend_col_gap)
        legend_y = legend_top + row_in_col * legend_row_height
        title_text = f"R{row_index + 1}  {state.row_labels[row_index]}"
        prompt_text = _truncate_text(state.row_prompts[row_index], 56)
        _render_text(surface, title_text, row_font, text_color, legend_x, legend_y)
        _render_text(
            surface,
            prompt_text,
            row_font,
            detail_color,
            legend_x,
            legend_y + row_font.get_height() + 2,
        )

    panel_gap = int(round(width * 0.018))
    panel_width = int(
        round((width - left_pad * 2 - panel_gap * (GRID_COLUMNS - 1)) / GRID_COLUMNS)
    )
    panel_y = legend_top + legend_row_height * 3 + int(round(height * 0.05))
    panel_height = max(
        int(round(height * 0.22)),
        height - panel_y - int(round(height * 0.14)),
    )

    for column_index in range(GRID_COLUMNS):
        panel_x = left_pad + column_index * (panel_width + panel_gap)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        is_active = column_index in state.active_columns
        panel_fill = FRONT_PANEL_ACTIVE if is_active else FRONT_PANEL
        panel_border = FRONT_PANEL_ACTIVE_BORDER if is_active else FRONT_PANEL_BORDER
        pygame.draw.rect(surface, panel_fill, panel_rect, border_radius=18)
        pygame.draw.rect(surface, panel_border, panel_rect, width=3, border_radius=18)

        label_surface = panel_title_font.render(
            f"Slot {column_index + 1}", True, TEXT_MAIN if is_active else TEXT_MUTED
        )
        current_token = state.current_style_tokens[column_index]
        value_surface = panel_value_font.render(
            str(current_token),
            True,
            FLOOR_SELECTED_BORDER if is_active else TEXT_ACCENT,
        )
        selected_row = state.selected_rows[column_index]
        status_text = (
            f"{_row_label(selected_row)} {'active' if is_active else 'locked'}"
        )
        status_surface = panel_status_font.render(status_text, True, TEXT_MUTED)
        prompt_surface = panel_status_font.render(
            state.row_labels[selected_row],
            True,
            TEXT_MUTED,
        )

        label_rect = label_surface.get_rect(center=(panel_rect.centerx, panel_rect.y + 36))
        value_rect = value_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery - 26))
        status_rect = status_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery + 16))
        prompt_rect = prompt_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery + 42))

        surface.blit(label_surface, label_rect)
        surface.blit(value_surface, value_rect)
        surface.blit(status_surface, status_rect)
        surface.blit(prompt_surface, prompt_rect)

        active_sample = state.column_drivers.get(column_index)
        if active_sample is not None:
            coord_text = f"x={active_sample.x:+.2f} z={active_sample.z:+.2f}"
        else:
            coord_text = "no person"
        coord_surface = panel_status_font.render(coord_text, True, TEXT_MUTED)
        coord_rect = coord_surface.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 30))
        surface.blit(coord_surface, coord_rect)

    footer_text = "Columns use position[2], rows use position[0], both clamped to [-3.5, 3.5]."
    footer_surface = subtitle_font.render(footer_text, True, TEXT_MUTED)
    footer_rect = footer_surface.get_rect(
        center=(width // 2, height - int(round(height * 0.08)))
    )
    surface.blit(footer_surface, footer_rect)

    if state.prompt_edit_active:
        overlay = pygame.Surface(size, pygame.SRCALPHA)
        overlay.fill(PROMPT_EDIT_OVERLAY)
        surface.blit(overlay, (0, 0))

        panel_rect = pygame.Rect(
            max(32, int(round(width * 0.09))),
            max(24, int(round(height * 0.12))),
            min(width - 64, int(round(width * 0.82))),
            min(height - 48, int(round(height * 0.44))),
        )
        pygame.draw.rect(
            surface,
            PROMPT_EDIT_PANEL_ACTIVE,
            panel_rect,
            border_radius=22,
        )
        pygame.draw.rect(
            surface,
            PROMPT_EDIT_PANEL_ACTIVE_BORDER,
            panel_rect,
            width=4,
            border_radius=22,
        )

        header_y = panel_rect.top + 20
        _render_text(
            surface,
            "Edit Seed Prompt",
            panel_title_font,
            TEXT_MAIN,
            panel_rect.left + 24,
            header_y,
        )
        _render_text(
            surface,
            "Enter applies and recomputes neighbors. Esc cancels. R resets context after you apply.",
            row_font,
            TEXT_MUTED,
            panel_rect.left + 24,
            header_y + panel_title_font.get_height() + 6,
        )

        editor_rect = pygame.Rect(
            panel_rect.left + 20,
            header_y + panel_title_font.get_height() + row_font.get_height() + 24,
            panel_rect.width - 40,
            panel_rect.height - panel_title_font.get_height() - row_font.get_height() - 68,
        )
        pygame.draw.rect(surface, PROMPT_EDIT_PANEL, editor_rect, border_radius=16)
        pygame.draw.rect(
            surface,
            PROMPT_EDIT_PANEL_BORDER,
            editor_rect,
            width=2,
            border_radius=16,
        )

        draft_prompt = state.prompt_edit_text if state.prompt_edit_text else state.prompt_bank_name
        wrapped_lines = _wrap_text(f"{draft_prompt}|", row_font, editor_rect.width - 28)
        text_y = editor_rect.top + 16
        for line in wrapped_lines[:6]:
            _render_text(
                surface,
                line,
                row_font,
                TEXT_MAIN,
                editor_rect.left + 14,
                text_y,
            )
            text_y += row_font.get_height() + 6
    return surface


def build_frames(
    canvas_sizes: Dict[str, Size],
    floor_canvas: str,
    front_canvas: str,
    state: GridState,
    *,
    show_reset_button: bool = False,
) -> Dict[str, pygame.Surface]:
    if floor_canvas not in canvas_sizes:
        raise SystemExit(
            f"Missing floor canvas {floor_canvas!r}; available={sorted(canvas_sizes)}"
        )
    frames: Dict[str, pygame.Surface] = {
        floor_canvas: create_floor_frame(canvas_sizes[floor_canvas], state),
    }
    if front_canvas in canvas_sizes:
        frames[front_canvas] = create_front_frame(
            canvas_sizes[front_canvas],
            state,
            show_reset_button=show_reset_button,
        )
    return frames


def _build_emulation_layout(
    canvas_sizes: Dict[str, Size],
    floor_canvas: str,
    front_canvas: str,
) -> EmulationLayout:
    floor_size = canvas_sizes[floor_canvas]
    front_size = canvas_sizes.get(front_canvas)
    display_info = pygame.display.Info()
    max_width = max(960, int(round(display_info.current_w * 0.9)))
    max_height = max(720, int(round(display_info.current_h * 0.9)))

    raw_width = floor_size[0] + EMULATION_MARGIN * 2
    if front_size is not None:
        raw_width += EMULATION_GAP + front_size[0]
    raw_height = max(floor_size[1], front_size[1] if front_size is not None else 0) + EMULATION_MARGIN * 2

    scale = min(max_width / raw_width, max_height / raw_height, 1.0)
    scale = max(scale, 0.2)
    margin = max(12, int(round(EMULATION_MARGIN * scale)))
    gap = max(12, int(round(EMULATION_GAP * scale))) if front_size is not None else 0
    floor_render = (
        max(1, int(round(floor_size[0] * scale))),
        max(1, int(round(floor_size[1] * scale))),
    )
    front_render = None
    if front_size is not None:
        front_render = (
            max(1, int(round(front_size[0] * scale))),
            max(1, int(round(front_size[1] * scale))),
        )

    content_height = max(floor_render[1], front_render[1] if front_render is not None else 0)
    window_width = floor_render[0] + margin * 2 + (gap + front_render[0] if front_render is not None else 0)
    window_height = content_height + margin * 2

    floor_rect = pygame.Rect(
        margin,
        margin + (content_height - floor_render[1]) // 2,
        floor_render[0],
        floor_render[1],
    )
    front_rect = None
    if front_render is not None:
        front_rect = pygame.Rect(
            floor_rect.right + gap,
            margin + (content_height - front_render[1]) // 2,
            front_render[0],
            front_render[1],
        )
    return EmulationLayout(
        window_size=(window_width, window_height),
        floor_rect=floor_rect,
        front_rect=front_rect,
    )


def _render_emulation_window(
    display_surface: pygame.Surface,
    layout: EmulationLayout,
    frames: Dict[str, pygame.Surface],
    floor_canvas: str,
    front_canvas: str,
) -> None:
    display_surface.fill(EMULATION_BG)
    floor_surface = frames[floor_canvas]
    floor_scaled = pygame.transform.smoothscale(floor_surface, layout.floor_rect.size)
    display_surface.blit(floor_scaled, layout.floor_rect)

    if layout.front_rect is not None and front_canvas in frames:
        front_surface = frames[front_canvas]
        front_scaled = pygame.transform.smoothscale(front_surface, layout.front_rect.size)
        display_surface.blit(front_scaled, layout.front_rect)

    pygame.display.flip()


def save_previews(
    preview_dir: Path,
    frames: Dict[str, pygame.Surface],
    canvas_sizes: Dict[str, Size],
    floor_canvas: str,
    front_canvas: str,
    state: GridState,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    for canvas_name, surface in frames.items():
        pygame.image.save(surface, preview_dir / f"{canvas_name}_canvas.png")

    metadata = {
        "floor_canvas": floor_canvas,
        "front_canvas": front_canvas if front_canvas in frames else None,
        "canvas_sizes": {name: [size[0], size[1]] for name, size in sorted(canvas_sizes.items())},
        "world_bounds": {"x": [WORLD_MIN, WORLD_MAX], "z": [WORLD_MIN, WORLD_MAX]},
        "grid": {"columns": GRID_COLUMNS, "rows": GRID_ROWS},
        "selected_rows": [row + 1 for row in state.selected_rows],
        "grid_tokens": state.grid_tokens,
        "prompt_bank_name": state.prompt_bank_name,
        "row_labels": state.row_labels,
        "row_prompts": state.row_prompts,
        "row_style_tokens": state.row_style_tokens,
        "current_style_tokens": state.current_style_tokens,
        "generation_kwargs": state.generation_kwargs,
        "magenta_server_url": state.magenta_server_url,
        "magenta_status": state.magenta_status,
        "active_columns": sorted(column + 1 for column in state.active_columns),
        "people": [
            {
                "person_id": sample.person_id,
                "x": sample.x,
                "z": sample.z,
                "column": sample.column_index + 1,
                "row": sample.row_index + 1,
                "confidence": sample.confidence,
            }
            for sample in state.people
        ],
        "people_count": state.people_count,
        "frame_index": state.frame_index,
        "pose_status": state.pose_status,
    }
    (preview_dir / "preview_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _connect_magenta_controller(
    args: argparse.Namespace,
    *,
    start_playback: bool,
) -> MagentaController:
    del start_playback
    try:
        return MagentaController(
            server_url=args.magenta_server_url,
            player_control_url=args.player_control_url,
            seed_prompt=args.music_prompt or args.seed_prompt,
            token_bank_path=args.token_bank_path,
            skip_nearest=args.skip_nearest,
            decode_window=args.decode_window,
            context_frames=args.context_frames,
            crossfade_frames=args.crossfade_frames,
            temperature=args.temperature,
            topk=args.topk,
            guidance_weight=args.guidance_weight,
        )
    except Exception as exc:
        raise SystemExit(f"Could not start Magenta control: {exc}") from exc


def _resolve_pose_port(protocol: str, requested_port: int | None) -> int:
    if requested_port is not None:
        return int(requested_port)
    if protocol == "lucid":
        return DEFAULT_LUCID_POSE_PORT
    return DEFAULT_ZMQ_POSE_PORT


def _connect_pose_source(args: argparse.Namespace) -> PoseSource | None:
    if not args.pose_endpoint:
        return None
    port = _resolve_pose_port(args.pose_protocol, args.pose_port)
    if args.pose_protocol == "lucid":
        return LucidPoseSource(
            endpoint=args.pose_endpoint,
            port=port,
            timeout=args.pose_timeout,
            poll_interval=args.pose_poll_interval,
            service_name=args.pose_service,
        )
    return ZmqPoseStreamSource(
        endpoint=args.pose_endpoint,
        port=port,
        timeout=args.pose_timeout,
        poll_interval=args.pose_poll_interval,
    )


def _apply_debug_people_if_any(state: GridState, args: argparse.Namespace) -> List[int]:
    debug_samples = _parse_debug_people(args.debug_person)
    if not debug_samples:
        return []
    return _apply_samples_to_state(
        state,
        debug_samples,
        pose_status="debug people",
        frame_index=None,
        timestamp_iso=None,
    )


def _apply_pose_message_if_any(
    state: GridState,
    message,
    source_label: str,
) -> List[int]:
    if message is None:
        return []
    samples = _extract_floor_samples_from_message(message)
    return _apply_samples_to_state(
        state,
        samples,
        pose_status=source_label,
        frame_index=_message_value(message, "frame_index", None),
        timestamp_iso=_message_value(message, "timestamp_iso", None),
    )


def _cursor_to_floor_point(
    floor_rect: pygame.Rect,
    mouse_position: Tuple[int, int],
) -> Tuple[float, float] | None:
    if floor_rect.width <= 0 or floor_rect.height <= 0:
        return None
    if not floor_rect.collidepoint(mouse_position):
        return None

    x_norm = (mouse_position[0] - floor_rect.left) / float(floor_rect.width)
    y_norm = (mouse_position[1] - floor_rect.top) / float(floor_rect.height)
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    world_x = WORLD_MIN + x_norm * (WORLD_MAX - WORLD_MIN)
    world_z = WORLD_MAX - y_norm * (WORLD_MAX - WORLD_MIN)
    return world_x, world_z


def _apply_emulated_cursor(
    state: GridState,
    *,
    mouse_down: bool,
    mouse_position: Tuple[int, int],
    click_position: Tuple[int, int] | None,
    floor_rect: pygame.Rect,
) -> List[int]:
    target_position = mouse_position if mouse_down else click_position
    if target_position is None:
        return _apply_samples_to_state(
            state,
            [],
            pose_status="emulate cursor ready",
            frame_index=None,
            timestamp_iso=None,
        )

    floor_point = _cursor_to_floor_point(floor_rect, target_position)
    if floor_point is None:
        return _apply_samples_to_state(
            state,
            [],
            pose_status="emulate cursor off-floor",
            frame_index=None,
            timestamp_iso=None,
        )

    sample = _sample_from_floor_point(
        person_id=1,
        x=floor_point[0],
        z=floor_point[1],
        confidence=1.0,
    )
    samples = [sample] if sample is not None else []
    pose_status = "emulate cursor active" if mouse_down else "emulate cursor click"
    return _apply_samples_to_state(
        state,
        samples,
        pose_status=pose_status,
        frame_index=None,
        timestamp_iso=None,
    )


def _sync_magenta_tokens(
    state: GridState,
    magenta_controller: MagentaController | None,
) -> bool:
    next_tokens = _selected_style_tokens(state)
    if next_tokens == state.current_style_tokens:
        return False
    if magenta_controller is not None:
        state.current_style_tokens = list(
            magenta_controller.update_style_tokens(next_tokens)
        )
        state.magenta_status = magenta_controller.status_text()
    else:
        state.current_style_tokens = next_tokens
    return True


def _load_seed_prompt_state(
    state: GridState,
    magenta_controller: MagentaController,
) -> None:
    state.prompt_bank_name = magenta_controller.seed_prompt
    state.row_labels = list(magenta_controller.row_labels)
    state.row_prompts = list(magenta_controller.row_prompts)
    state.row_style_tokens = [
        list(tokens) for tokens in magenta_controller.row_style_tokens
    ]
    state.grid_tokens = _build_token_columns(
        state.row_style_tokens,
    )


def _start_prompt_edit(state: GridState) -> None:
    state.prompt_edit_active = True
    state.prompt_edit_text = state.prompt_bank_name
    state.prompt_edit_status = "Editing seed prompt. Enter applies; Esc cancels."


def _cancel_prompt_edit(state: GridState) -> None:
    state.prompt_edit_active = False
    state.prompt_edit_text = state.prompt_bank_name
    state.prompt_edit_status = "Seed prompt edit canceled. Press P to edit again."


def _apply_seed_prompt_edit(
    state: GridState,
    magenta_controller: MagentaController | None,
) -> bool:
    if magenta_controller is None:
        state.prompt_edit_status = "Seed prompt update unavailable: no magenta controller."
        return False
    try:
        magenta_controller.reseed_prompt(state.prompt_edit_text)
        _load_seed_prompt_state(state, magenta_controller)
        state.prompt_edit_text = state.prompt_bank_name
        state.prompt_edit_active = False
        state.prompt_edit_status = "Seed prompt updated. Press P to edit again."
        return True
    except Exception as exc:
        state.prompt_edit_status = f"Seed prompt update failed: {exc}"
        return False


def _reset_magenta_context(
    state: GridState,
    magenta_controller: MagentaController | None,
) -> None:
    if magenta_controller is None:
        state.magenta_status = "context reset unavailable: no magenta controller"
        return
    try:
        state.current_style_tokens = list(
            magenta_controller.reset_context(_selected_style_tokens(state))
        )
        state.magenta_status = magenta_controller.status_text()
    except Exception as exc:
        state.magenta_status = f"context reset failed: {exc}"


def _visual_state_signature(state: GridState) -> Tuple[object, ...]:
    people_signature = tuple(
        (
            sample.person_id,
            round(sample.x, 3),
            round(sample.z, 3),
            sample.column_index,
            sample.row_index,
        )
        for sample in state.people
    )
    return (
        tuple(state.selected_rows),
        tuple(tuple(column) for column in state.grid_tokens),
        tuple(sorted(state.active_columns)),
        people_signature,
        state.prompt_bank_name,
        tuple(state.row_labels),
        tuple(state.row_prompts),
        tuple(state.current_style_tokens),
        state.magenta_status,
        state.pose_status,
        state.people_count,
        state.prompt_edit_active,
        state.prompt_edit_text,
        state.prompt_edit_status,
    )


def main() -> None:
    args = parse_args()

    pygame.init()
    pygame.mixer.quit()
    pygame.font.init()

    canvas_sizes = _effective_canvas_sizes(args.config)
    if not canvas_sizes:
        raise SystemExit("No canvases defined in room config.")

    floor_canvas = _pick_floor_canvas(canvas_sizes, args.floor_canvas)
    state = GridState()
    magenta_controller = None
    pose_source = None
    emulate_layout = None
    emulate_window = None

    try:
        magenta_controller = _connect_magenta_controller(
            args,
            start_playback=not args.preview_only,
        )
        _load_seed_prompt_state(state, magenta_controller)
        state.magenta_server_url = magenta_controller.server_url
        state.current_style_tokens = list(magenta_controller.current_style_tokens)
        state.generation_kwargs = dict(magenta_controller.generation_kwargs)
        state.magenta_status = magenta_controller.status_text()
        state.prompt_edit_text = state.prompt_bank_name

        if _apply_debug_people_if_any(state, args):
            _sync_magenta_tokens(state, magenta_controller)

        if args.emulate:
            state.pose_status = "emulate cursor ready"
        else:
            pose_source = _connect_pose_source(args)
        if pose_source is not None:
            bootstrap_message = pose_source.wait_for_latest(args.pose_bootstrap_seconds)
            if bootstrap_message is not None:
                changed_columns = _apply_pose_message_if_any(
                    state,
                    bootstrap_message,
                    pose_source.label,
                )
                if changed_columns:
                    _sync_magenta_tokens(state, magenta_controller)
            elif state.pose_status == "static rows":
                state.pose_status = f"{pose_source.label} waiting"

        frames = build_frames(
            canvas_sizes,
            floor_canvas=floor_canvas,
            front_canvas=args.front_canvas,
            state=state,
            show_reset_button=args.emulate,
        )

        if args.preview_dir:
            save_previews(
                Path(args.preview_dir).expanduser(),
                frames,
                canvas_sizes,
                floor_canvas=floor_canvas,
                front_canvas=args.front_canvas,
                state=state,
            )

        if args.preview_only:
            return

        router = None
        if args.emulate:
            emulate_layout = _build_emulation_layout(
                canvas_sizes,
                floor_canvas=floor_canvas,
                front_canvas=args.front_canvas,
            )
            emulate_window = pygame.display.set_mode(emulate_layout.window_size)
            pygame.display.set_caption("music-floor emulate")
        else:
            router = CanvasProjectorRouter.from_room_config(
                args.config,
                caption_prefix="music-floor: ",
                auto_display=True,
                renderer_kwargs={"use_sdl2_window": True},
            )

        clock = pygame.time.Clock()
        last_visual_signature = None
        running = True
        emulate_mouse_down = False
        emulate_click_position = None
        reset_requested = False
        prompt_apply_requested = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    if state.prompt_edit_active:
                        _cancel_prompt_edit(state)
                        pygame.key.stop_text_input()
                        continue
                    running = False
                elif state.prompt_edit_active and event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        prompt_apply_requested = True
                    elif event.key == pygame.K_BACKSPACE:
                        state.prompt_edit_text = state.prompt_edit_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        _cancel_prompt_edit(state)
                        pygame.key.stop_text_input()
                elif state.prompt_edit_active and event.type == pygame.TEXTINPUT:
                    state.prompt_edit_text += event.text
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    _start_prompt_edit(state)
                    pygame.key.start_text_input()
                if args.emulate:
                    if state.prompt_edit_active:
                        continue
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if (
                            emulate_layout is not None
                            and _emulation_prompt_button_hit(
                                layout=emulate_layout,
                                front_size=canvas_sizes[args.front_canvas],
                                mouse_position=event.pos,
                            )
                        ):
                            _start_prompt_edit(state)
                            pygame.key.start_text_input()
                        elif (
                            emulate_layout is not None
                            and _emulation_reset_button_hit(
                                layout=emulate_layout,
                                front_size=canvas_sizes[args.front_canvas],
                                mouse_position=event.pos,
                            )
                        ):
                            reset_requested = True
                        else:
                            emulate_mouse_down = True
                            emulate_click_position = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        emulate_mouse_down = False

            if reset_requested:
                _reset_magenta_context(state, magenta_controller)
                reset_requested = False

            if prompt_apply_requested:
                if _apply_seed_prompt_edit(state, magenta_controller):
                    _sync_magenta_tokens(state, magenta_controller)
                    pygame.key.stop_text_input()
                prompt_apply_requested = False

            if state.prompt_edit_active:
                if magenta_controller is not None:
                    magenta_controller.poll()
                    state.magenta_status = magenta_controller.status_text()
                visual_signature = _visual_state_signature(state)
                if visual_signature != last_visual_signature:
                    frames = build_frames(
                        canvas_sizes,
                        floor_canvas=floor_canvas,
                        front_canvas=args.front_canvas,
                        state=state,
                        show_reset_button=args.emulate,
                    )
                    if args.emulate and emulate_window is not None and emulate_layout is not None:
                        _render_emulation_window(
                            emulate_window,
                            emulate_layout,
                            frames,
                            floor_canvas=floor_canvas,
                            front_canvas=args.front_canvas,
                        )
                    elif router is not None:
                        router.render(
                            frames,
                            display=True,
                            block=False,
                            use_optimized=args.use_optimized,
                            sync_display=True,
                        )
                    last_visual_signature = visual_signature
                clock.tick(args.fps)
                continue

            if args.emulate and emulate_layout is not None:
                changed_columns = _apply_emulated_cursor(
                    state,
                    mouse_down=emulate_mouse_down,
                    mouse_position=pygame.mouse.get_pos(),
                    click_position=emulate_click_position,
                    floor_rect=emulate_layout.floor_rect,
                )
                emulate_click_position = None
                if changed_columns:
                    _sync_magenta_tokens(state, magenta_controller)
            elif pose_source is not None:
                latest_message = pose_source.poll_latest()
                if latest_message is not None:
                    changed_columns = _apply_pose_message_if_any(
                        state,
                        latest_message,
                        pose_source.label,
                    )
                    if changed_columns:
                        _sync_magenta_tokens(state, magenta_controller)

            if magenta_controller is not None:
                magenta_controller.poll()
                magenta_status = magenta_controller.status_text()
                current_style_tokens = list(magenta_controller.current_style_tokens)
                if magenta_status != state.magenta_status:
                    state.magenta_status = magenta_status
                if current_style_tokens != state.current_style_tokens:
                    state.current_style_tokens = current_style_tokens
            visual_signature = _visual_state_signature(state)
            if visual_signature != last_visual_signature:
                frames = build_frames(
                    canvas_sizes,
                    floor_canvas=floor_canvas,
                    front_canvas=args.front_canvas,
                    state=state,
                    show_reset_button=args.emulate,
                )
                if args.emulate and emulate_window is not None and emulate_layout is not None:
                    _render_emulation_window(
                        emulate_window,
                        emulate_layout,
                        frames,
                        floor_canvas=floor_canvas,
                        front_canvas=args.front_canvas,
                    )
                elif router is not None:
                    router.render(
                        frames,
                        display=True,
                        block=False,
                        use_optimized=args.use_optimized,
                        sync_display=True,
                    )
                last_visual_signature = visual_signature
            clock.tick(args.fps)
    finally:
        pygame.key.stop_text_input()
        if pose_source is not None:
            pose_source.close()
        if magenta_controller is not None:
            magenta_controller.close()
        pygame.quit()


if __name__ == "__main__":
    main()
