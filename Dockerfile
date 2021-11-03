FROM ubuntu:20.04

ARG DEPS="\
    python3 \
    python3-pip \
    python3-opencv \
"

ADD . /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install ${DEPS} -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install -r requirements.txt