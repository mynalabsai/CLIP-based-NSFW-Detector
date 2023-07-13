FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV ccap=75

RUN apt-get update -y && \
    apt-get install -y python3-pip

ADD requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /code

CMD uvicorn api:app --host 0.0.0.0 --port 80 --log-level debug
