# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y git
RUN pip install -q --upgrade pip

RUN pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

RUN pip install --upgrade "jax[cpu]"
RUN pip install -q git+https://www.github.com/google/neural-tangents

RUN pip install pandas
