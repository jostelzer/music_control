#!/usr/bin/env python3
"""Render Music Floor with persistent per-column selections and live pose input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from music_floor_prompt_banks import (
    DEFAULT_PROMPT_BANK,
    PromptRow,
    format_prompt_bank_listing,
    resolve_prompt_rows,
)


def _prepend_path(path: Path, package_name: str) -> bool:
    if not (path / package_name).is_dir():
        return False
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return True


def _ensure_projection_mapping_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root.parent / "projector-mapping",
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
    repo_root = Path(__file__).resolve().parent
    pose_candidates = [
        repo_root.parent / "3d_pose_estimation_optitrack" / "src",
        Path.home() / "git" / "3d_pose_estimation_optitrack" / "src",
    ]
    if not any(_prepend_path(candidate, "pose_stream") for candidate in pose_candidates):
        raise SystemExit(
            "Could not locate the sibling '3d_pose_estimation_optitrack' repo. "
            "Expected it at ../3d_pose_estimation_optitrack or ~/git/3d_pose_estimation_optitrack."
        )

    lunar_candidates = [
        repo_root.parent / "lunar_tools",
        Path.home() / "git" / "lunar_tools",
        repo_root.parent / "lunar_tools_refact",
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

BLACK = (0, 0, 0)
TEXT_MAIN = (235, 235, 235)
TEXT_MUTED = (160, 160, 160)
TEXT_ACCENT = (205, 205, 205)

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
    prompt_bank_name: str = DEFAULT_PROMPT_BANK
    magenta_server_url: str = DEFAULT_MAGENTA_SERVER_URL
    magenta_status: str = "magenta idle"


class PoseStreamSource:
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


class MagentaController:
    def __init__(
        self,
        server_url: str,
        prompt_rows: List[PromptRow],
    ) -> None:
        if len(prompt_rows) != GRID_ROWS:
            raise ValueError(
                f"Expected {GRID_ROWS} prompt rows, got {len(prompt_rows)}."
            )
        self.prompt_rows = [
            PromptRow(label=row.label.strip(), prompt=row.prompt.strip())
            for row in prompt_rows
        ]
        self.server_url = self._normalize_server_url(server_url)
        self.row_style_tokens: List[List[int]] = [
            self._tokenize_style_text(row.prompt) for row in self.prompt_rows
        ]
        self.current_style_tokens: List[int] = list(self.row_style_tokens[DEFAULT_ROW_INDEX])
        self._latest_status = "server control idle"
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

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
    ):
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib_request.Request(
            f"{self.server_url}{path}",
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

    def _tokenize_style_text(self, prompt: str) -> List[int]:
        stream_info = self._request_json("GET", "/stream_info")
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

    def _apply_control(self, style_tokens: List[int]) -> None:
        response = self._request_json(
            "POST",
            "/control",
            data=json.dumps({"style_tokens": [int(token) for token in style_tokens]}).encode(
                "utf-8"
            ),
            content_type="application/json",
        )
        if isinstance(response, dict) and isinstance(response.get("style_tokens"), list):
            self.current_style_tokens = [int(token) for token in response["style_tokens"]]
        else:
            self.current_style_tokens = [int(token) for token in style_tokens]
        session_is_active = bool(response.get("session_is_active")) if isinstance(response, dict) else False
        playback_state = str(response.get("playback_state") or "unknown") if isinstance(response, dict) else "unknown"
        session_text = "session active" if session_is_active else "no player session"
        self._latest_status = (
            f"server control {session_text} | playback={playback_state.lower()} @ {self.server_url}"
        )

    def poll(self) -> None:
        return None

    def update_style_tokens(self, style_tokens: List[int]) -> List[int]:
        tokens = [int(token) for token in style_tokens]
        self._apply_control(tokens)
        return list(self.current_style_tokens)

    def status_text(self) -> str:
        return self._latest_status

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Floor demo with live ankle-driven Magenta style-token selection."
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
        "--pose-endpoint",
        type=str,
        default="iki",
        help="ZMQ pose stream hostname/IP.",
    )
    parser.add_argument(
        "--pose-port",
        type=int,
        default=5570,
        help="Pose stream TCP port.",
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
        "--prompt-bank",
        type=str,
        default=DEFAULT_PROMPT_BANK,
        help="Built-in six-row prompt bank for the floor grid.",
    )
    parser.add_argument(
        "--row-prompt",
        action="append",
        default=[],
        help="Override a row prompt. Repeat up to six times; overrides rows from the top.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Optional text or JSON file defining exactly six prompt rows.",
    )
    parser.add_argument(
        "--list-prompt-banks",
        action="store_true",
        help="Print the built-in prompt banks and exit.",
    )
    parser.add_argument(
        "--music-prompt",
        type=str,
        default=None,
        help="Legacy alias for overriding row 1 only.",
    )
    parser.add_argument(
        "--magenta-server-url",
        type=str,
        default=DEFAULT_MAGENTA_SERVER_URL,
        help="URL of the Magenta realtime server.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Deprecated; prompt banks no longer use random token rows.",
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


def _extract_position_from_keypoint(keypoint) -> Tuple[float, float, float] | None:
    position = getattr(keypoint, "position", None)
    if not isinstance(position, tuple) or len(position) != 3:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError):
        return None


def _extract_floor_samples_from_message(message) -> List[PersonFloorSample]:
    samples: List[PersonFloorSample] = []
    for person in getattr(message, "people", []):
        ankle_positions: List[Tuple[float, float, float]] = []
        ankle_confidences: List[float] = []
        for keypoint_name in ("left_ankle", "right_ankle"):
            keypoint = person.keypoints.get(keypoint_name)
            if keypoint is None:
                continue
            position = _extract_position_from_keypoint(keypoint)
            if position is None:
                continue
            ankle_positions.append(position)
            confidence = getattr(keypoint, "confidence", None)
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
            person_id=int(getattr(person, "id", 0)),
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


def create_front_frame(size: Size, state: GridState) -> pygame.Surface:
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
        "Each row is a full six-token prompt. Matching all columns to one row recreates that prompt exactly.",
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
        f"Prompt bank: {state.prompt_bank_name}",
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 2 + 38,
    )
    _render_text(
        surface,
        f"Live style tokens: {state.current_style_tokens}",
        subtitle_font,
        TEXT_ACCENT,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 3 + 50,
    )
    _render_text(
        surface,
        state.magenta_status,
        subtitle_font,
        TEXT_MUTED,
        left_pad,
        top_pad + title_font.get_height() + subtitle_font.get_height() * 4 + 62,
    )

    legend_top = top_pad + title_font.get_height() + subtitle_font.get_height() * 5 + 82
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
    return surface


def build_frames(
    canvas_sizes: Dict[str, Size],
    floor_canvas: str,
    front_canvas: str,
    state: GridState,
) -> Dict[str, pygame.Surface]:
    if floor_canvas not in canvas_sizes:
        raise SystemExit(
            f"Missing floor canvas {floor_canvas!r}; available={sorted(canvas_sizes)}"
        )
    frames: Dict[str, pygame.Surface] = {
        floor_canvas: create_floor_frame(canvas_sizes[floor_canvas], state),
    }
    if front_canvas in canvas_sizes:
        frames[front_canvas] = create_front_frame(canvas_sizes[front_canvas], state)
    return frames


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
    prompt_rows: List[PromptRow],
    *,
    start_playback: bool,
) -> MagentaController:
    del start_playback
    try:
        return MagentaController(
            server_url=args.magenta_server_url,
            prompt_rows=prompt_rows,
        )
    except Exception as exc:
        raise SystemExit(f"Could not start Magenta control: {exc}") from exc


def _connect_pose_source(args: argparse.Namespace) -> PoseStreamSource | None:
    if not args.pose_endpoint:
        return None
    return PoseStreamSource(
        endpoint=args.pose_endpoint,
        port=args.pose_port,
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
    endpoint: str,
    port: int,
) -> List[int]:
    if message is None:
        return []
    samples = _extract_floor_samples_from_message(message)
    return _apply_samples_to_state(
        state,
        samples,
        pose_status=f"pose {endpoint}:{port}",
        frame_index=getattr(message, "frame_index", None),
        timestamp_iso=getattr(message, "timestamp_iso", None),
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
        tuple(state.current_style_tokens),
        state.magenta_status,
        state.pose_status,
        state.people_count,
    )


def main() -> None:
    args = parse_args()
    if args.list_prompt_banks:
        print(format_prompt_bank_listing())
        return

    prompt_bank_name, prompt_rows = resolve_prompt_rows(
        prompt_bank=args.prompt_bank,
        prompt_file=args.prompt_file,
        row_prompts=args.row_prompt,
        legacy_music_prompt=args.music_prompt,
    )

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

    try:
        magenta_controller = _connect_magenta_controller(
            args,
            prompt_rows=prompt_rows,
            start_playback=not args.preview_only,
        )
        state.prompt_bank_name = prompt_bank_name
        state.row_labels = [row.label for row in prompt_rows]
        state.row_prompts = [row.prompt for row in prompt_rows]
        state.magenta_server_url = magenta_controller.server_url
        state.row_style_tokens = [
            list(tokens) for tokens in magenta_controller.row_style_tokens
        ]
        state.current_style_tokens = list(magenta_controller.current_style_tokens)
        state.grid_tokens = _build_token_columns(
            state.row_style_tokens,
        )
        state.magenta_status = magenta_controller.status_text()

        if _apply_debug_people_if_any(state, args):
            _sync_magenta_tokens(state, magenta_controller)

        pose_source = _connect_pose_source(args)
        if pose_source is not None:
            bootstrap_message = pose_source.wait_for_latest(args.pose_bootstrap_seconds)
            if bootstrap_message is not None:
                changed_columns = _apply_pose_message_if_any(
                    state,
                    bootstrap_message,
                    args.pose_endpoint,
                    args.pose_port,
                )
                if changed_columns:
                    _sync_magenta_tokens(state, magenta_controller)
            elif state.pose_status == "static rows":
                state.pose_status = f"pose {args.pose_endpoint}:{args.pose_port} waiting"

        frames = build_frames(
            canvas_sizes,
            floor_canvas=floor_canvas,
            front_canvas=args.front_canvas,
            state=state,
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

        router = CanvasProjectorRouter.from_room_config(
            args.config,
            caption_prefix="music-floor: ",
            auto_display=True,
            renderer_kwargs={"use_sdl2_window": True},
        )

        clock = pygame.time.Clock()
        last_visual_signature = None
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            if pose_source is not None:
                latest_message = pose_source.poll_latest()
                if latest_message is not None:
                    changed_columns = _apply_pose_message_if_any(
                        state,
                        latest_message,
                        args.pose_endpoint,
                        args.pose_port,
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
                )
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
        if pose_source is not None:
            pose_source.close()
        if magenta_controller is not None:
            magenta_controller.close()
        pygame.quit()


if __name__ == "__main__":
    main()
