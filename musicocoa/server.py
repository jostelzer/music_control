#!/usr/bin/env python3

"""Runs the repo-local MusicCoCa prompt embedding service."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from aiohttp import web

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.service import MusicCoCaLab
else:
  from .service import MusicCoCaLab


class MusicCoCaServer:
  """HTTP wrapper around the repo-local MusicCoCaLab."""

  def __init__(
      self,
      *,
      backend_name: str,
      device: str,
      runtime_style_token_depth: int,
      workers: int,
      disable_tf32: bool,
  ):
    self._lab = MusicCoCaLab(
        backend_name=backend_name,
        device=device,
        runtime_style_token_depth=runtime_style_token_depth,
        workers=workers,
        disable_tf32=disable_tf32,
    )
    self._app = web.Application()
    self._app.router.add_get('/health', self.handle_health)
    self._app.router.add_post('/embed', self.handle_embed)
    self._app.router.add_post('/embed_batch', self.handle_embed_batch)
    self._app.on_cleanup.append(self._on_cleanup)

  async def _on_cleanup(self, app: web.Application) -> None:
    del app
    self._lab.close()

  async def _read_json(self, request: web.Request) -> dict[str, Any]:
    try:
      body = await request.json()
    except json.JSONDecodeError as exc:
      raise web.HTTPBadRequest(text=f'invalid JSON body: {exc}') from exc
    if not isinstance(body, dict):
      raise web.HTTPBadRequest(text='expected JSON object body')
    return body

  async def handle_health(self, request: web.Request) -> web.Response:
    del request
    return web.json_response(self._lab.health())

  async def handle_embed(self, request: web.Request) -> web.Response:
    body = await self._read_json(request)
    try:
      result = self._lab.embed_text(
          str(body.get('text', '')),
          include_embedding=bool(body.get('include_embedding')),
          include_full_tokens=bool(body.get('include_full_tokens')),
      )
    except (TypeError, ValueError) as exc:
      raise web.HTTPBadRequest(text=str(exc)) from exc
    except RuntimeError as exc:
      raise web.HTTPServiceUnavailable(text=str(exc)) from exc
    return web.json_response(result)

  async def handle_embed_batch(self, request: web.Request) -> web.Response:
    body = await self._read_json(request)
    texts = body.get('texts', [])
    try:
      items = self._lab.embed_batch_text(
          list(texts),
          include_embedding=bool(body.get('include_embedding')),
          include_full_tokens=bool(body.get('include_full_tokens')),
      )
    except (TypeError, ValueError) as exc:
      raise web.HTTPBadRequest(text=str(exc)) from exc
    except RuntimeError as exc:
      raise web.HTTPServiceUnavailable(text=str(exc)) from exc
    return web.json_response({'count': len(items), 'items': items})

  def run(self, *, host: str, port: int) -> None:
    web.run_app(self._app, host=host, port=port)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=8770)
  parser.add_argument('--backend', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--disable-tf32', action='store_true')
  args = parser.parse_args()
  MusicCoCaServer(
      backend_name=args.backend,
      device=args.device,
      runtime_style_token_depth=args.runtime_style_token_depth,
      workers=args.workers,
      disable_tf32=args.disable_tf32,
  ).run(host=args.host, port=args.port)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
