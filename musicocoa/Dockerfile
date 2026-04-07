ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:25.02-tf2-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAGENTA_REALTIME_REPO=/opt/magenta-realtime \
    MAGENTA_RT_CACHE_DIR=/opt/magenta-rt-cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tensorflow==2.20.0 \
    aiohttp \
    absl-py \
    google-cloud-storage \
    huggingface-hub \
    requests \
    sentencepiece \
    tqdm

WORKDIR /opt/music_control

COPY music_control/musicocoa /opt/music_control/musicocoa
COPY magenta-realtime/magenta_rt /opt/magenta-realtime/magenta_rt

RUN mkdir -p /opt/magenta-rt-cache && \
    chmod +x /opt/music_control/musicocoa/docker_entrypoint.sh

ENV PYTHONPATH=/opt/music_control:/opt/magenta-realtime

EXPOSE 8773

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/music_control/musicocoa/docker_entrypoint.sh"]
CMD []
