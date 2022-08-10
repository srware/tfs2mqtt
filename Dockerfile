FROM ubuntu:22.04

ARG DEPS="\
    python3 \
    python3-pip \
"

COPY *.py *.txt /app/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install ${DEPS} -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements-docker.txt && \
    python3 -m pip install ovmsclient --no-deps
