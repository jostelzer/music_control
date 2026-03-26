#!/usr/bin/env python3
"""Render Music Floor with persistent per-column selections and live pose input."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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

import pygame

from projection_mapping import CanvasProjectorRouter
from projection_mapping.apply_correction import ProjectionAlignment
from projection_mapping.multi import load_room_layout

Size = Tuple[int, int]

WORLD_MIN = -3.5
WORLD_MAX = 3.5
GRID_COLUMNS = 6
GRID_ROWS = 6
DEFAULT_ROW_INDEX = 3

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
    active_columns: set[int] = field(default_factory=set)
    people: List[PersonFloorSample] = field(default_factory=list)
    column_drivers: Dict[int, PersonFloorSample] = field(default_factory=dict)
    people_count: int = 0
    frame_index: int | None = None
    timestamp_iso: str | None = None
    pose_status: str = "static rows"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Floor demo with live ankle-driven grid selection."
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
) -> None:
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

    for column_index, sample in best_by_column.items():
        state.selected_rows[column_index] = sample.row_index
    state.column_drivers = best_by_column


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


def create_floor_frame(size: Size, state: GridState) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill(BLACK)
    width, height = size

    x_centers = _grid_centers(GRID_COLUMNS)
    y_centers = list(reversed(_grid_centers(GRID_ROWS)))
    cell_w = width / GRID_COLUMNS
    cell_h = height / GRID_ROWS
    radius = max(8, int(round(min(cell_w, cell_h) * 0.24)))
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
    subtitle_font = _font(max(20, height // 22))
    panel_title_font = _font(max(18, height // 20))
    panel_value_font = _font(max(24, height // 14))
    panel_status_font = _font(max(16, height // 28))

    top_pad = int(round(height * 0.07))
    left_pad = int(round(width * 0.06))
    _render_text(surface, "Music Floor", title_font, TEXT_MAIN, left_pad, top_pad)
    _render_text(
        surface,
        "Ankle midpoint controls the nearest row. Each column keeps its last locked green row.",
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

    panel_gap = int(round(width * 0.018))
    panel_width = int(
        round((width - left_pad * 2 - panel_gap * (GRID_COLUMNS - 1)) / GRID_COLUMNS)
    )
    panel_height = int(round(height * 0.40))
    panel_y = int(round(height * 0.38))

    for column_index in range(GRID_COLUMNS):
        panel_x = left_pad + column_index * (panel_width + panel_gap)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        is_active = column_index in state.active_columns
        panel_fill = FRONT_PANEL_ACTIVE if is_active else FRONT_PANEL
        panel_border = FRONT_PANEL_ACTIVE_BORDER if is_active else FRONT_PANEL_BORDER
        pygame.draw.rect(surface, panel_fill, panel_rect, border_radius=18)
        pygame.draw.rect(surface, panel_border, panel_rect, width=3, border_radius=18)

        label_surface = panel_title_font.render(
            f"Column {column_index + 1}", True, TEXT_MAIN if is_active else TEXT_MUTED
        )
        value_surface = panel_value_font.render(
            _row_label(state.selected_rows[column_index]),
            True,
            FLOOR_SELECTED_BORDER if is_active else TEXT_ACCENT,
        )
        status_text = "active" if is_active else "locked"
        status_surface = panel_status_font.render(status_text, True, TEXT_MUTED)

        label_rect = label_surface.get_rect(center=(panel_rect.centerx, panel_rect.y + 36))
        value_rect = value_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery - 12))
        status_rect = status_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery + 42))

        surface.blit(label_surface, label_rect)
        surface.blit(value_surface, value_rect)
        surface.blit(status_surface, status_rect)

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


def _connect_pose_source(args: argparse.Namespace) -> PoseStreamSource | None:
    if not args.pose_endpoint:
        return None
    return PoseStreamSource(
        endpoint=args.pose_endpoint,
        port=args.pose_port,
        timeout=args.pose_timeout,
        poll_interval=args.pose_poll_interval,
    )


def _apply_debug_people_if_any(state: GridState, args: argparse.Namespace) -> None:
    debug_samples = _parse_debug_people(args.debug_person)
    if not debug_samples:
        return
    _apply_samples_to_state(
        state,
        debug_samples,
        pose_status="debug people",
        frame_index=None,
        timestamp_iso=None,
    )


def _apply_pose_message_if_any(state: GridState, message, endpoint: str, port: int) -> None:
    if message is None:
        return
    samples = _extract_floor_samples_from_message(message)
    _apply_samples_to_state(
        state,
        samples,
        pose_status=f"pose {endpoint}:{port}",
        frame_index=getattr(message, "frame_index", None),
        timestamp_iso=getattr(message, "timestamp_iso", None),
    )


def main() -> None:
    args = parse_args()

    pygame.init()
    pygame.font.init()

    canvas_sizes = _effective_canvas_sizes(args.config)
    if not canvas_sizes:
        raise SystemExit("No canvases defined in room config.")

    floor_canvas = _pick_floor_canvas(canvas_sizes, args.floor_canvas)
    state = GridState()
    _apply_debug_people_if_any(state, args)

    pose_source = None
    try:
        pose_source = _connect_pose_source(args)
        if pose_source is not None:
            bootstrap_message = pose_source.wait_for_latest(args.pose_bootstrap_seconds)
            if bootstrap_message is not None:
                _apply_pose_message_if_any(
                    state,
                    bootstrap_message,
                    args.pose_endpoint,
                    args.pose_port,
                )
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
                    _apply_pose_message_if_any(
                        state,
                        latest_message,
                        args.pose_endpoint,
                        args.pose_port,
                    )

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
            clock.tick(args.fps)
    finally:
        if pose_source is not None:
            pose_source.close()
        pygame.quit()


if __name__ == "__main__":
    main()
