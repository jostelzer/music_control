#!/usr/bin/env python3
"""Render Music Floor with compositional prompt recombination."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from floor_slots import music_floor_demo as base

pygame = base.pygame

PROMPT_FIELD_COUNT = 6
PROMPT_OPTION_COUNT = 6
DEFAULT_PROMPT_ANCHOR = "minimal techno"
DEFAULT_PROMPT_SCHEMA_NAME = "prompt_recombiner_v1"


@dataclass(frozen=True)
class SlotChoice:
    label: str
    prompt: str


@dataclass(frozen=True)
class PromptField:
    name: str
    choices: Tuple[SlotChoice, ...]


PROMPT_FIELDS: Tuple[PromptField, ...] = (
    PromptField(
        name="Beat",
        choices=(
            SlotChoice("Four Floor", "steady four on the floor"),
            SlotChoice("Drive", "driving pulse"),
            SlotChoice("Roll", "rolling syncopation"),
            SlotChoice("Slow Burn", "slow-burning groove"),
            SlotChoice("Low Slung", "low-slung groove"),
            SlotChoice("Stomp", "straight warehouse stomp"),
        ),
    ),
    PromptField(
        name="Bass",
        choices=(
            SlotChoice("Sub Roll", "rolling sub bass"),
            SlotChoice("Rubber", "rubbery bassline"),
            SlotChoice("Acid", "subtle acid bass"),
            SlotChoice("Mono Pulse", "pulsing mono bass"),
            SlotChoice("Drone", "sub drone"),
            SlotChoice("Hollow", "hollow low-end throb"),
        ),
    ),
    PromptField(
        name="Perc",
        choices=(
            SlotChoice("Clicks", "clicky percussion"),
            SlotChoice("Hats", "shuffled hats"),
            SlotChoice("Rim Ticks", "dry rim ticks"),
            SlotChoice("Metallic", "metallic hat chatter"),
            SlotChoice("Dub Echoes", "dubby percussion echoes"),
            SlotChoice("Claps", "snappy clap accents"),
        ),
    ),
    PromptField(
        name="Texture",
        choices=(
            SlotChoice("Sequencer", "hypnotic sequencer"),
            SlotChoice("Stab", "filtered synth stab"),
            SlotChoice("Dub Chords", "muted dub chords"),
            SlotChoice("Blips", "minimal synth blips"),
            SlotChoice("Pad Smear", "detuned pad smear"),
            SlotChoice("Acid Frags", "acid line fragments"),
        ),
    ),
    PromptField(
        name="Space",
        choices=(
            SlotChoice("Punchy", "tight punchy compression"),
            SlotChoice("Hi-Fi", "clean hi-fi sheen"),
            SlotChoice("Analog", "warm analog blur"),
            SlotChoice("Reverb", "lush reverb tail"),
            SlotChoice("Warehouse", "dry warehouse room"),
            SlotChoice("Studio", "polished studio mix"),
        ),
    ),
    PromptField(
        name="Mood",
        choices=(
            SlotChoice("Hypnotic", "hypnotic, minor key tension"),
            SlotChoice("Brooding", "brooding, open fifth drones"),
            SlotChoice("Chromatic", "mysterious, chromatic movement"),
            SlotChoice("Focused", "focused, modal restraint"),
            SlotChoice("Suspended", "playful, suspended harmonies"),
            SlotChoice("Pressure", "dark, low-register pressure"),
        ),
    ),
)

SCHEMA_NAME = DEFAULT_PROMPT_SCHEMA_NAME


def _validated_rows(selected_rows: Sequence[int]) -> List[int]:
    rows = [int(index) for index in selected_rows]
    if len(rows) != PROMPT_FIELD_COUNT:
        raise ValueError(
            f"Expected {PROMPT_FIELD_COUNT} selected rows, got {len(rows)}."
        )
    for field_index, row_index in enumerate(rows):
        if row_index < 0 or row_index >= PROMPT_OPTION_COUNT:
            raise ValueError(
                f"Field {field_index + 1} row must be between 0 and "
                f"{PROMPT_OPTION_COUNT - 1}; got {row_index}."
            )
    return rows


def field_names() -> List[str]:
    return [field.name for field in PROMPT_FIELDS]


def field_choice_labels() -> List[List[str]]:
    return [[choice.label for choice in field.choices] for field in PROMPT_FIELDS]


def field_choice_prompts() -> List[List[str]]:
    return [[choice.prompt for choice in field.choices] for field in PROMPT_FIELDS]


def selected_choice_labels(selected_rows: Sequence[int]) -> List[str]:
    rows = _validated_rows(selected_rows)
    return [
        field.choices[row_index].label
        for field, row_index in zip(PROMPT_FIELDS, rows)
    ]


def selected_choice_prompts(selected_rows: Sequence[int]) -> List[str]:
    rows = _validated_rows(selected_rows)
    return [
        field.choices[row_index].prompt
        for field, row_index in zip(PROMPT_FIELDS, rows)
    ]


def compose_prompt(
    selected_rows: Sequence[int],
    *,
    anchor: str = DEFAULT_PROMPT_ANCHOR,
) -> str:
    normalized_anchor = " ".join((anchor or "").strip().split())
    if not normalized_anchor:
        raise ValueError("Prompt anchor cannot be empty.")
    parts = [normalized_anchor, *selected_choice_prompts(selected_rows)]
    return ", ".join(part for part in parts if part)


def format_schema(
    *,
    anchor: str = DEFAULT_PROMPT_ANCHOR,
) -> str:
    lines = [
        f"schema: {DEFAULT_PROMPT_SCHEMA_NAME}",
        f"anchor: {anchor}",
    ]
    for field_index, field in enumerate(PROMPT_FIELDS, start=1):
        lines.append(f"{field_index}. {field.name}")
        for row_index, choice in enumerate(field.choices, start=1):
            lines.append(f"   R{row_index}: {choice.label} :: {choice.prompt}")
    return "\n".join(lines)


@dataclass
class GridState:
    selected_rows: List[int] = field(
        default_factory=lambda: [base.DEFAULT_ROW_INDEX for _ in range(base.GRID_COLUMNS)]
    )
    current_style_tokens: List[int] = field(
        default_factory=lambda: [0 for _ in range(base.GRID_COLUMNS)]
    )
    active_columns: set[int] = field(default_factory=set)
    people: List[base.PersonFloorSample] = field(default_factory=list)
    column_drivers: Dict[int, base.PersonFloorSample] = field(default_factory=dict)
    people_count: int = 0
    frame_index: int | None = None
    timestamp_iso: str | None = None
    pose_status: str = "static rows"
    prompt_schema_name: str = SCHEMA_NAME
    prompt_anchor: str = DEFAULT_PROMPT_ANCHOR
    current_prompt_text: str = ""
    magenta_server_url: str = base.DEFAULT_MAGENTA_SERVER_URL
    magenta_status: str = "magenta idle"
    field_labels: List[str] = field(default_factory=field_names)
    option_labels_by_column: List[List[str]] = field(default_factory=field_choice_labels)
    option_prompts_by_column: List[List[str]] = field(default_factory=field_choice_prompts)


class PromptRecombinerController:
    def __init__(
        self,
        server_url: str,
        player_control_url: str | None,
        initial_prompt: str,
    ) -> None:
        self.server_url = self._normalize_server_url(server_url)
        self.player_control_url = self._normalize_optional_url(player_control_url)
        self._use_player_control = False
        self.current_prompt_text = self._normalize_prompt(initial_prompt)
        self.current_style_tokens = self._tokenize_style_text(self.current_prompt_text)
        self._latest_status = "server control idle"
        self._refresh_player_control_mode()
        self._apply_prompt(self.current_prompt_text, self.current_style_tokens)

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        normalized = " ".join((prompt or "").strip().split())
        if not normalized:
            raise ValueError("Prompt text cannot be empty.")
        return normalized

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

    def _tokenize_style_text(self, prompt: str) -> List[int]:
        stream_info = self._request_json("GET", "/stream_info")
        token_count = int(stream_info.get("style_token_count", base.GRID_COLUMNS))
        if token_count != base.GRID_COLUMNS:
            raise RuntimeError(
                f"Magenta server expects {token_count} style tokens, "
                f"but Music Floor uses {base.GRID_COLUMNS} columns."
            )
        response = self._request_json(
            "POST",
            "/style_tokens",
            data=prompt.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )
        if not isinstance(response, list) or len(response) != base.GRID_COLUMNS:
            raise RuntimeError(
                f"Invalid /style_tokens response: expected {base.GRID_COLUMNS} tokens."
            )
        return [int(token) for token in response]

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
        status_text = str(response.get("status_text") or "").strip()
        if status_text:
            self._latest_status = status_text
        self._use_player_control = True

    def _consume_control_response(
        self,
        response,
        fallback_tokens: Sequence[int],
        fallback_prompt: str,
        *,
        status_prefix: str,
    ) -> None:
        if isinstance(response, dict) and isinstance(response.get("style_tokens"), list):
            self.current_style_tokens = [int(token) for token in response["style_tokens"]]
        else:
            self.current_style_tokens = [int(token) for token in fallback_tokens]
        self.current_prompt_text = fallback_prompt
        if isinstance(response, dict):
            status_text = str(response.get("status_text") or "").strip()
            if status_text:
                self._latest_status = status_text
                return
        session_is_active = bool(response.get("session_is_active")) if isinstance(response, dict) else False
        playback_state = (
            str(response.get("playback_state") or "unknown")
            if isinstance(response, dict)
            else "unknown"
        )
        session_text = "session active" if session_is_active else "no player session"
        self._latest_status = (
            f"{status_prefix} {session_text} | playback={playback_state.lower()} @ {self.server_url}"
        )

    def _apply_prompt(
        self,
        prompt_text: str,
        style_tokens: Sequence[int] | None = None,
        *,
        reset_context: bool = False,
    ) -> None:
        prompt = self._normalize_prompt(prompt_text)
        tokens = (
            [int(token) for token in style_tokens]
            if style_tokens is not None
            else self._tokenize_style_text(prompt)
        )
        self._refresh_player_control_mode()
        if self._use_player_control and self.player_control_url is not None:
            payload = {
                "style_tokens": tokens,
                "style_text": prompt,
                "style_source": "prompt_recombiner",
            }
            path = "/reset_session" if reset_context else "/update_style_tokens"
            response = self._request_json(
                "POST",
                path,
                data=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                base_url=self.player_control_url,
            )
            status_prefix = "player reset" if reset_context else "player control"
        else:
            response = self._request_json(
                "POST",
                "/control",
                data=json.dumps({"style_tokens": tokens}).encode("utf-8"),
                content_type="application/json",
            )
            status_prefix = "server control"
            if reset_context:
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
            prompt,
            status_prefix=status_prefix,
        )

    def poll(self) -> None:
        return None

    def update_prompt(self, prompt_text: str) -> List[int]:
        self._apply_prompt(prompt_text)
        return list(self.current_style_tokens)

    def reset_context(self, prompt_text: str | None = None) -> List[int]:
        prompt = self.current_prompt_text if prompt_text is None else prompt_text
        self._apply_prompt(prompt, reset_context=True)
        return list(self.current_style_tokens)

    def status_text(self) -> str:
        return self._latest_status

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Floor demo with per-column prompt recombination."
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
        default=base.DEFAULT_POSE_PROTOCOL,
        help="Pose transport to use. Defaults to Lucid HTTP.",
    )
    parser.add_argument(
        "--pose-service",
        type=str,
        default=base.DEFAULT_LUCID_POSE_SERVICE,
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
        "--prompt-anchor",
        type=str,
        default=DEFAULT_PROMPT_ANCHOR,
        help="Fixed prefix prepended before the six selected prompt fields.",
    )
    parser.add_argument(
        "--list-schema",
        action="store_true",
        help="Print the compositional prompt schema and exit.",
    )
    parser.add_argument(
        "--magenta-server-url",
        type=str,
        default=base.DEFAULT_MAGENTA_SERVER_URL,
        help="URL of the Magenta realtime server.",
    )
    parser.add_argument(
        "--player-control-url",
        type=str,
        default=base.DEFAULT_PLAYER_CONTROL_URL,
        help="Local music_floor_player control URL. Empty string disables it and falls back to direct server control.",
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


def _wrap_text(
    text: str,
    font: pygame.font.Font,
    max_width: int,
    *,
    max_lines: int,
) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    words = cleaned.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if not current or font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and " ".join(lines) != cleaned:
        last_line = lines[-1].rstrip(". ")
        while last_line and font.size(f"{last_line}…")[0] > max_width:
            last_line = last_line[:-1].rstrip()
        lines[-1] = f"{last_line}…" if last_line else "…"
    return lines


def _selected_choice_prompts_for_state(state: GridState) -> List[str]:
    return [
        state.option_prompts_by_column[column_index][row_index]
        for column_index, row_index in enumerate(state.selected_rows)
    ]


def _compose_state_prompt(state: GridState) -> str:
    return compose_prompt(state.selected_rows, anchor=state.prompt_anchor)


def create_floor_frame(size: base.Size, state: GridState) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill(base.BLACK)
    width, height = size

    x_centers = base._grid_centers(base.GRID_COLUMNS)
    y_centers = list(reversed(base._grid_centers(base.GRID_ROWS)))
    cell_w = width / base.GRID_COLUMNS
    cell_h = height / base.GRID_ROWS
    radius = max(8, int(round(min(cell_w, cell_h) * 0.24)))
    option_font = base._font(max(12, int(round(radius * 0.78))))
    lane_width = max(24, int(round(cell_w * 0.64)))
    lane_top = int(round(cell_h * 0.35))
    lane_height = max(1, height - lane_top * 2)
    lane_border_width = max(2, int(round(radius * 0.08)))
    separator_width = max(2, int(round(cell_w * 0.03)))

    for column_index, x_world in enumerate(x_centers):
        x_px, _ = base._world_to_pixel(size, x_world, 0.0)
        lane_rect = pygame.Rect(0, 0, lane_width, lane_height)
        lane_rect.center = (x_px, height // 2)
        is_active = column_index in state.active_columns
        lane_fill = base.FLOOR_ACTIVE_FILL if is_active else base.FLOOR_COLUMN_FILL
        lane_border_color = (
            base.FLOOR_ACTIVE_BORDER if is_active else base.FLOOR_COLUMN_BORDER
        )
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
            width=lane_border_width,
            border_radius=max(12, lane_width // 2),
        )

    for column_index in range(base.GRID_COLUMNS - 1):
        boundary_world = (x_centers[column_index] + x_centers[column_index + 1]) / 2.0
        x_px, _ = base._world_to_pixel(size, boundary_world, 0.0)
        pygame.draw.line(
            surface,
            base.FLOOR_SEPARATOR,
            (x_px, int(round(cell_h * 0.25))),
            (x_px, height - int(round(cell_h * 0.25))),
            separator_width,
        )

    for column_index, x_world in enumerate(x_centers):
        selected_row = state.selected_rows[column_index]
        for row_index, z_world in enumerate(y_centers):
            center = base._world_to_pixel(size, x_world, z_world)
            is_selected = row_index == selected_row
            fill = base.FLOOR_SELECTED_FILL if is_selected else base.FLOOR_CIRCLE_FILL
            border = base.FLOOR_SELECTED_BORDER if is_selected else base.FLOOR_CIRCLE_BORDER
            pygame.draw.circle(surface, fill, center, radius)
            pygame.draw.circle(
                surface,
                border,
                center,
                radius,
                width=max(2, int(round(radius * 0.08))),
            )
            option_surface = option_font.render(str(row_index + 1), True, base.FLOOR_TOKEN_TEXT)
            option_rect = option_surface.get_rect(center=center)
            surface.blit(option_surface, option_rect)

    marker_radius = max(6, int(round(min(cell_w, cell_h) * 0.08)))
    for sample in state.people:
        marker_center = base._world_to_pixel(size, sample.x, sample.z)
        pygame.draw.circle(surface, base.FLOOR_MARKER, marker_center, marker_radius, width=2)

    return surface


def create_front_frame(
    size: base.Size,
    state: GridState,
    *,
    show_reset_button: bool = False,
) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill(base.BLACK)
    width, height = size

    title_font = base._font(max(28, height // 12))
    subtitle_font = base._font(max(18, height // 24))
    panel_title_font = base._font(max(18, height // 24))
    panel_value_font = base._font(max(22, height // 17))
    panel_status_font = base._font(max(13, height // 34))
    option_font = base._font(max(12, height // 40))

    top_pad = int(round(height * 0.06))
    left_pad = int(round(width * 0.05))
    max_text_width = width - left_pad * 2
    reset_button_rect = base._front_reset_button_rect(size) if show_reset_button else None
    header_max_width = max_text_width
    if reset_button_rect is not None:
        header_max_width = max(
            240,
            reset_button_rect.left - left_pad - 24,
        )

    base._render_text(
        surface,
        "Music Floor Prompt Recombiner",
        title_font,
        base.TEXT_MAIN,
        left_pad,
        top_pad,
    )
    subtitle_lines = _wrap_text(
        "Each column picks one prompt field. The current six choices are recomposed into one prompt.",
        subtitle_font,
        header_max_width,
        max_lines=2,
    )
    subtitle_y = top_pad + title_font.get_height() + 10
    for line_index, line in enumerate(subtitle_lines):
        base._render_text(
            surface,
            line,
            subtitle_font,
            base.TEXT_MUTED,
            left_pad,
            subtitle_y + line_index * (subtitle_font.get_height() + 4),
        )

    frame_label = (
        f"people={state.people_count} frame={state.frame_index}"
        if state.frame_index is not None
        else f"people={state.people_count}"
    )
    subtitle_block_height = max(1, len(subtitle_lines)) * (subtitle_font.get_height() + 4)
    meta_y = top_pad + title_font.get_height() + subtitle_block_height + 20
    base._render_text(
        surface,
        f"{state.pose_status}  |  {frame_label}",
        subtitle_font,
        base.TEXT_MUTED,
        left_pad,
        meta_y,
    )
    base._render_text(
        surface,
        f"Schema: {state.prompt_schema_name}  |  Anchor: {state.prompt_anchor}",
        subtitle_font,
        base.TEXT_MUTED,
        left_pad,
        meta_y + subtitle_font.get_height() + 10,
    )

    prompt_title_y = meta_y + subtitle_font.get_height() * 2 + 28
    base._render_text(surface, "Composed prompt", subtitle_font, base.TEXT_ACCENT, left_pad, prompt_title_y)
    prompt_lines = _wrap_text(
        state.current_prompt_text,
        subtitle_font,
        max_text_width,
        max_lines=3,
    )
    for line_index, line in enumerate(prompt_lines):
        base._render_text(
            surface,
            line,
            subtitle_font,
            base.TEXT_MAIN,
            left_pad,
            prompt_title_y + subtitle_font.get_height() + 8 + line_index * (subtitle_font.get_height() + 4),
        )

    prompt_block_height = max(1, len(prompt_lines)) * (subtitle_font.get_height() + 4)
    info_y = prompt_title_y + subtitle_font.get_height() + prompt_block_height + 18
    base._render_text(
        surface,
        f"Live style tokens: {state.current_style_tokens}",
        subtitle_font,
        base.TEXT_ACCENT,
        left_pad,
        info_y,
    )
    status_lines = _wrap_text(
        state.magenta_status,
        subtitle_font,
        max_text_width,
        max_lines=2,
    )
    for line_index, line in enumerate(status_lines):
        base._render_text(
            surface,
            line,
            subtitle_font,
            base.TEXT_MUTED,
            left_pad,
            info_y + subtitle_font.get_height() + 10 + line_index * (subtitle_font.get_height() + 3),
        )

    if reset_button_rect is not None:
        pygame.draw.rect(surface, base.FRONT_BUTTON_FILL, reset_button_rect, border_radius=16)
        pygame.draw.rect(
            surface,
            base.FRONT_BUTTON_BORDER,
            reset_button_rect,
            width=3,
            border_radius=16,
        )
        reset_label_surface = panel_title_font.render("Reset Context", True, base.FRONT_BUTTON_TEXT)
        reset_label_rect = reset_label_surface.get_rect(
            center=(reset_button_rect.centerx, reset_button_rect.centery - 8)
        )
        surface.blit(reset_label_surface, reset_label_rect)
        reset_hint_surface = panel_status_font.render("Press R", True, base.FRONT_BUTTON_HINT)
        reset_hint_rect = reset_hint_surface.get_rect(
            center=(reset_button_rect.centerx, reset_button_rect.centery + 16)
        )
        surface.blit(reset_hint_surface, reset_hint_rect)

    panel_gap = int(round(width * 0.012))
    panel_width = int(round((width - left_pad * 2 - panel_gap * (base.GRID_COLUMNS - 1)) / base.GRID_COLUMNS))
    panel_y = info_y + subtitle_font.get_height() * (2 + len(status_lines)) + 34
    panel_height = max(
        int(round(height * 0.34)),
        height - panel_y - int(round(height * 0.10)),
    )
    option_row_gap = 6
    option_area_top = 90
    option_area_bottom = 38
    option_row_height = max(
        24,
        int((panel_height - option_area_top - option_area_bottom - option_row_gap * (base.GRID_ROWS - 1)) / base.GRID_ROWS),
    )

    for column_index in range(base.GRID_COLUMNS):
        panel_x = left_pad + column_index * (panel_width + panel_gap)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        is_active = column_index in state.active_columns
        panel_fill = base.FRONT_PANEL_ACTIVE if is_active else base.FRONT_PANEL
        panel_border = base.FRONT_PANEL_ACTIVE_BORDER if is_active else base.FRONT_PANEL_BORDER
        pygame.draw.rect(surface, panel_fill, panel_rect, border_radius=18)
        pygame.draw.rect(surface, panel_border, panel_rect, width=3, border_radius=18)

        field_surface = panel_title_font.render(
            state.field_labels[column_index],
            True,
            base.TEXT_MAIN if is_active else base.TEXT_MUTED,
        )
        selected_row = state.selected_rows[column_index]
        token_surface = panel_value_font.render(
            f"R{selected_row + 1}",
            True,
            base.FLOOR_SELECTED_BORDER if is_active else base.TEXT_ACCENT,
        )
        field_rect = field_surface.get_rect(center=(panel_rect.centerx, panel_rect.y + 28))
        token_rect = token_surface.get_rect(center=(panel_rect.centerx, panel_rect.y + 60))
        surface.blit(field_surface, field_rect)
        surface.blit(token_surface, token_rect)

        for row_index in range(base.GRID_ROWS):
            row_rect = pygame.Rect(
                panel_rect.x + 10,
                panel_rect.y + option_area_top + row_index * (option_row_height + option_row_gap),
                panel_rect.width - 20,
                option_row_height,
            )
            is_selected = row_index == selected_row
            row_fill = base.FLOOR_ACTIVE_FILL if is_selected else (32, 32, 32)
            row_border = base.FLOOR_ACTIVE_BORDER if is_selected else (72, 72, 72)
            pygame.draw.rect(surface, row_fill, row_rect, border_radius=10)
            pygame.draw.rect(surface, row_border, row_rect, width=2, border_radius=10)

            row_code_surface = option_font.render(
                f"R{row_index + 1}",
                True,
                base.TEXT_MAIN if is_selected else base.TEXT_MUTED,
            )
            label_text = base._truncate_text(
                state.option_labels_by_column[column_index][row_index],
                12,
            )
            label_surface = option_font.render(
                label_text,
                True,
                base.FLOOR_SELECTED_BORDER if is_selected else base.TEXT_MAIN,
            )
            surface.blit(row_code_surface, (row_rect.x + 8, row_rect.y + 6))
            surface.blit(label_surface, (row_rect.x + 34, row_rect.y + 6))

        phrase_text = base._truncate_text(
            state.option_prompts_by_column[column_index][selected_row],
            20,
        )
        phrase_surface = panel_status_font.render(phrase_text, True, base.TEXT_MUTED)
        phrase_rect = phrase_surface.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 36))
        surface.blit(phrase_surface, phrase_rect)

        active_sample = state.column_drivers.get(column_index)
        coord_text = (
            f"x={active_sample.x:+.2f} z={active_sample.z:+.2f}"
            if active_sample is not None
            else "no person"
        )
        coord_surface = panel_status_font.render(coord_text, True, base.TEXT_MUTED)
        coord_rect = coord_surface.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 18))
        surface.blit(coord_surface, coord_rect)

    footer_text = "Columns use position[2], rows use position[0], both clamped to [-3.5, 3.5]."
    footer_surface = subtitle_font.render(footer_text, True, base.TEXT_MUTED)
    footer_rect = footer_surface.get_rect(center=(width // 2, height - int(round(height * 0.05))))
    surface.blit(footer_surface, footer_rect)
    return surface


def build_frames(
    canvas_sizes: Dict[str, base.Size],
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


def save_previews(
    preview_dir: Path,
    frames: Dict[str, pygame.Surface],
    canvas_sizes: Dict[str, base.Size],
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
        "world_bounds": {"x": [base.WORLD_MIN, base.WORLD_MAX], "z": [base.WORLD_MIN, base.WORLD_MAX]},
        "grid": {"columns": base.GRID_COLUMNS, "rows": base.GRID_ROWS},
        "selected_rows": [row + 1 for row in state.selected_rows],
        "field_labels": state.field_labels,
        "selected_choice_labels": selected_choice_labels(state.selected_rows),
        "selected_choice_prompts": _selected_choice_prompts_for_state(state),
        "current_prompt_text": state.current_prompt_text,
        "prompt_schema_name": state.prompt_schema_name,
        "prompt_anchor": state.prompt_anchor,
        "fields": [
            {
                "column": index + 1,
                "name": state.field_labels[index],
                "choices": [
                    {
                        "row": row_index + 1,
                        "label": label,
                        "prompt": prompt,
                    }
                    for row_index, (label, prompt) in enumerate(
                        zip(
                            state.option_labels_by_column[index],
                            state.option_prompts_by_column[index],
                        )
                    )
                ],
            }
            for index in range(base.GRID_COLUMNS)
        ],
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
    prompt_text: str,
    *,
    start_playback: bool,
) -> PromptRecombinerController:
    del start_playback
    try:
        return PromptRecombinerController(
            server_url=args.magenta_server_url,
            player_control_url=args.player_control_url,
            initial_prompt=prompt_text,
        )
    except Exception as exc:
        raise SystemExit(f"Could not start Magenta control: {exc}") from exc


def _sync_magenta_prompt(
    state: GridState,
    magenta_controller: PromptRecombinerController | None,
) -> bool:
    next_prompt = _compose_state_prompt(state)
    if next_prompt == state.current_prompt_text:
        return False
    if magenta_controller is not None:
        state.current_style_tokens = list(magenta_controller.update_prompt(next_prompt))
        state.current_prompt_text = magenta_controller.current_prompt_text
        state.magenta_status = magenta_controller.status_text()
    else:
        state.current_prompt_text = next_prompt
    return True


def _reset_magenta_context(
    state: GridState,
    magenta_controller: PromptRecombinerController | None,
) -> None:
    if magenta_controller is None:
        state.magenta_status = "context reset unavailable: no magenta controller"
        return
    try:
        state.current_style_tokens = list(
            magenta_controller.reset_context(state.current_prompt_text)
        )
        state.current_prompt_text = magenta_controller.current_prompt_text
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
        tuple(sorted(state.active_columns)),
        people_signature,
        tuple(state.current_style_tokens),
        state.current_prompt_text,
        state.magenta_status,
        state.pose_status,
        state.people_count,
    )


def main() -> None:
    args = parse_args()
    if args.list_schema:
        print(format_schema(anchor=args.prompt_anchor))
        return

    pygame.init()
    pygame.mixer.quit()
    pygame.font.init()

    canvas_sizes = base._effective_canvas_sizes(args.config)
    if not canvas_sizes:
        raise SystemExit("No canvases defined in room config.")

    floor_canvas = base._pick_floor_canvas(canvas_sizes, args.floor_canvas)
    state = GridState()
    state.prompt_anchor = " ".join((args.prompt_anchor or "").strip().split())
    state.current_prompt_text = _compose_state_prompt(state)
    magenta_controller = None
    pose_source = None
    emulate_layout = None
    emulate_window = None

    try:
        magenta_controller = _connect_magenta_controller(
            args,
            prompt_text=state.current_prompt_text,
            start_playback=not args.preview_only,
        )
        state.prompt_schema_name = SCHEMA_NAME
        state.magenta_server_url = magenta_controller.server_url
        state.current_style_tokens = list(magenta_controller.current_style_tokens)
        state.current_prompt_text = magenta_controller.current_prompt_text
        state.magenta_status = magenta_controller.status_text()

        if base._apply_debug_people_if_any(state, args):
            _sync_magenta_prompt(state, magenta_controller)

        if args.emulate:
            state.pose_status = "emulate cursor ready"
        else:
            pose_source = base._connect_pose_source(args)
        if pose_source is not None:
            bootstrap_message = pose_source.wait_for_latest(args.pose_bootstrap_seconds)
            if bootstrap_message is not None:
                changed_columns = base._apply_pose_message_if_any(
                    state,
                    bootstrap_message,
                    pose_source.label,
                )
                if changed_columns:
                    _sync_magenta_prompt(state, magenta_controller)
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
            emulate_layout = base._build_emulation_layout(
                canvas_sizes,
                floor_canvas=floor_canvas,
                front_canvas=args.front_canvas,
            )
            emulate_window = pygame.display.set_mode(emulate_layout.window_size)
            pygame.display.set_caption("music-floor-prompt-recombiner emulate")
        else:
            router = base.CanvasProjectorRouter.from_room_config(
                args.config,
                caption_prefix="music-floor-prompt-recombiner: ",
                auto_display=True,
                renderer_kwargs={"use_sdl2_window": True},
            )

        clock = pygame.time.Clock()
        last_visual_signature = None
        running = True
        emulate_mouse_down = False
        emulate_click_position = None
        reset_requested = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_requested = True
                if args.emulate:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if (
                            emulate_layout is not None
                            and base._emulation_reset_button_hit(
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

            if args.emulate and emulate_layout is not None:
                changed_columns = base._apply_emulated_cursor(
                    state,
                    mouse_down=emulate_mouse_down,
                    mouse_position=pygame.mouse.get_pos(),
                    click_position=emulate_click_position,
                    floor_rect=emulate_layout.floor_rect,
                )
                emulate_click_position = None
                if changed_columns:
                    _sync_magenta_prompt(state, magenta_controller)
            elif pose_source is not None:
                latest_message = pose_source.poll_latest()
                if latest_message is not None:
                    changed_columns = base._apply_pose_message_if_any(
                        state,
                        latest_message,
                        pose_source.label,
                    )
                    if changed_columns:
                        _sync_magenta_prompt(state, magenta_controller)

            if magenta_controller is not None:
                magenta_controller.poll()
                magenta_status = magenta_controller.status_text()
                current_style_tokens = list(magenta_controller.current_style_tokens)
                if magenta_status != state.magenta_status:
                    state.magenta_status = magenta_status
                if current_style_tokens != state.current_style_tokens:
                    state.current_style_tokens = current_style_tokens
                if magenta_controller.current_prompt_text != state.current_prompt_text:
                    state.current_prompt_text = magenta_controller.current_prompt_text

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
                    base._render_emulation_window(
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
        if pose_source is not None:
            pose_source.close()
        if magenta_controller is not None:
            magenta_controller.close()
        pygame.quit()


if __name__ == "__main__":
    main()
