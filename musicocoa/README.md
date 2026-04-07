# MusicCoCa Deployment

`musicocoa` now has three simple entry points:

- Direct local run:
  `bash /Users/jjj/git/music_control/musicocoa/run_xla_server.sh --host 0.0.0.0 --port 8773`
- Local Docker run:
  `bash /Users/jjj/git/music_control/musicocoa/run_docker_server.sh --rebuild`
- Remote Docker spool:
  `bash /Users/jjj/git/music_control/musicocoa/spool_remote_docker.sh ias@iki`
- Stop local or remote Docker container:
  `bash /Users/jjj/git/music_control/musicocoa/stop_docker_server.sh --host ias@iki`

The Docker runners now wait for both:

- `GET /health`
- one real `POST /embed`

So a container is only considered ready once it can actually embed a prompt.

## Modes

- Fast mode is the default. It keeps TF32 enabled and is the mode that hit the `250 prompts / 250 ms` response budget in verification.
- Strict mode disables TF32 for tighter CUDA fidelity:
  `bash /Users/jjj/git/music_control/musicocoa/run_docker_server.sh --strict`
- All Docker entry points accept `--base-image <image>` if you need to override the default runtime stack.

## Docker Notes

The default Docker base image is:

- `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`

On top of that base, the image installs:

- `tensorflow==2.20.0`

That combination matters on Blackwell-class GPUs such as the `RTX 5090` on `iki`:

- the NGC base provides a CUDA stack that can actually register `sm_120`
- the newer TensorFlow wheel can deserialize and run the legacy `XlaCallModule`

The Docker image is built from a compact temporary context containing only:

- `music_control/musicocoa`
- `magenta-realtime/magenta_rt`

This keeps the Docker context small and avoids depending on a remote checkout.

The image fetches Magenta RT assets at runtime into `/opt/magenta-rt-cache`.

If a host exposes a GPU to Docker but TensorFlow cannot register it cleanly,
the `xla_exact` backend falls back to `/CPU:0` instead of crashing on the first
embed. This makes the same spool command usable on mixed machines such as `iki`.

## Remote Spool

`spool_remote_docker.sh` uses the local repos on this Mac as the source of truth:

1. It creates a temporary Docker build context locally.
2. It streams that context over SSH.
3. It builds the image on the remote host.
4. It starts the container bound to `127.0.0.1:<host_port>`.
5. It prints the SSH tunnel command to reach the service from this Mac.

Example strict-fidelity deploy to `iki`:

`bash /Users/jjj/git/music_control/musicocoa/spool_remote_docker.sh ias@iki --strict`

If you already know the target host should stay on CPU:

`bash /Users/jjj/git/music_control/musicocoa/spool_remote_docker.sh ias@iki --cpu`

If you want to leave the `iki` service up on the GPU path:

`bash /Users/jjj/git/music_control/musicocoa/spool_remote_docker.sh ias@iki --name musicocoa-iki-gpu --host-port 8774`
