#!/usr/bin/env python3
"""Gradio UI for blending two prompts and sending the result to Magenta control."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

import gradio as gr

from floor_slots.music_floor_prompt_banks import get_prompt_bank

DEFAULT_MAGENTA_SERVER_URL = "http://graciosa:8013"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_BLEND = 0.5

_DEFAULT_PROMPTS = get_prompt_bank("floor_journey")
DEFAULT_PROMPT_A = _DEFAULT_PROMPTS[0].prompt
DEFAULT_PROMPT_B = _DEFAULT_PROMPTS[3].prompt


class MagentaBlendController:
    def __init__(self, server_url: str) -> None:
        self.server_url = self._normalize_server_url(server_url)

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

    def tokenize_prompt(self, prompt: str) -> List[int]:
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Prompt text cannot be empty.")

        stream_info = self._request_json("GET", "/stream_info")
        token_count = int(stream_info.get("style_token_count", 0))
        if token_count <= 0:
            raise RuntimeError("Server did not report a valid style_token_count.")

        response = self._request_json(
            "POST",
            "/style_tokens",
            data=text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )
        if not isinstance(response, list) or len(response) != token_count:
            raise RuntimeError(
                f"Invalid /style_tokens response: expected {token_count} tokens."
            )
        return [int(token) for token in response]

    def embed_prompt(self, prompt: str) -> List[float]:
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Prompt text cannot be empty.")
        response = self._request_json(
            "POST",
            "/style",
            data=text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )
        if not isinstance(response, list) or not response:
            raise RuntimeError("Invalid /style response: expected a non-empty embedding.")
        return [float(value) for value in response]

    def quantize_embedding(self, embedding: Sequence[float]) -> List[int]:
        response = self._request_json(
            "POST",
            "/style_quantize",
            data=json.dumps({"embedding": [float(value) for value in embedding]}).encode("utf-8"),
            content_type="application/json",
        )
        if not isinstance(response, list) or not response:
            raise RuntimeError(
                "Invalid /style_quantize response: expected a non-empty token list."
            )
        return [int(token) for token in response]

    def apply_control(self, style_tokens: List[int]) -> dict:
        return self._request_json(
            "POST",
            "/control",
            data=json.dumps({"style_tokens": [int(token) for token in style_tokens]}).encode(
                "utf-8"
            ),
            content_type="application/json",
        )


def _normalize_embedding(embedding: Sequence[float]) -> List[float]:
    vector = [float(value) for value in embedding]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError("Embedding norm must be > 0.")
    return [value / norm for value in vector]


def blend_prompt_embeddings(
    prompt_a_embedding: Sequence[float],
    prompt_b_embedding: Sequence[float],
    blend: float,
) -> List[float]:
    if len(prompt_a_embedding) != len(prompt_b_embedding):
        raise ValueError("Prompt embeddings must have the same length.")

    blend_amount = min(1.0, max(0.0, float(blend)))
    normalized_a = _normalize_embedding(prompt_a_embedding)
    normalized_b = _normalize_embedding(prompt_b_embedding)
    blended = [
        (1.0 - blend_amount) * value_a + blend_amount * value_b
        for value_a, value_b in zip(normalized_a, normalized_b)
    ]
    return _normalize_embedding(blended)


def _format_control_status(server_url: str, control_response: dict, blend: float) -> str:
    session_is_active = bool(control_response.get("session_is_active"))
    playback_state = str(control_response.get("playback_state") or "unknown").lower()
    style_tokens = control_response.get("style_tokens")
    status_parts = [
        f"server={server_url}",
        f"blend={blend:.2f}",
        f"session_active={session_is_active}",
        f"playback={playback_state}",
    ]
    if isinstance(style_tokens, list):
        status_parts.append(f"applied_tokens={json.dumps([int(token) for token in style_tokens])}")
    return "\n".join(status_parts)


def run_blend(
    prompt_a: str,
    prompt_b: str,
    blend: float,
    server_url: str,
):
    controller = MagentaBlendController(server_url)
    prompt_a_tokens = controller.tokenize_prompt(prompt_a)
    prompt_b_tokens = controller.tokenize_prompt(prompt_b)
    prompt_a_embedding = controller.embed_prompt(prompt_a)
    prompt_b_embedding = controller.embed_prompt(prompt_b)
    blended_embedding = blend_prompt_embeddings(prompt_a_embedding, prompt_b_embedding, blend)
    blended_tokens = controller.quantize_embedding(blended_embedding)
    control_response = controller.apply_control(blended_tokens)
    return (
        _format_control_status(controller.server_url, control_response, blend),
        json.dumps(prompt_a_tokens),
        json.dumps(prompt_b_tokens),
        json.dumps(blended_tokens),
    )


def build_app(default_server_url: str = DEFAULT_MAGENTA_SERVER_URL) -> gr.Blocks:
    with gr.Blocks(title="Music Prompt Blending") as demo:
        gr.Markdown(
            """
            # Music Prompt Blending
            Blend two prompts with a slider, then press `Run` to interpolate in style
            embedding space and send the quantized result to the already-running
            Magenta player/server process.
            """
        )
        with gr.Row():
            server_url = gr.Textbox(
                label="Magenta Server URL",
                value=default_server_url,
            )
            blend = gr.Slider(
                label="Blend Toward Prompt B",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=DEFAULT_BLEND,
                info="0.0 = Prompt A only, 1.0 = Prompt B only",
            )
        prompt_a = gr.Textbox(
            label="Prompt A",
            value=DEFAULT_PROMPT_A,
            lines=4,
        )
        prompt_b = gr.Textbox(
            label="Prompt B",
            value=DEFAULT_PROMPT_B,
            lines=4,
        )
        run_button = gr.Button("Run", variant="primary")
        status = gr.Textbox(label="Status", lines=6)
        with gr.Row():
            prompt_a_tokens = gr.Textbox(label="Prompt A Tokens", lines=2)
            prompt_b_tokens = gr.Textbox(label="Prompt B Tokens", lines=2)
            blended_tokens = gr.Textbox(label="Blended Tokens", lines=2)

        run_button.click(
            fn=run_blend,
            inputs=[prompt_a, prompt_b, blend, server_url],
            outputs=[status, prompt_a_tokens, prompt_b_tokens, blended_tokens],
            api_name="run_blend",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gradio prompt blending UI for Music Floor.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_MAGENTA_SERVER_URL,
        help="URL of the Magenta realtime server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host interface for Gradio.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for Gradio.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(args.server_url)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
