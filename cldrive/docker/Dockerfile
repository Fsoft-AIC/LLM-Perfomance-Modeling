FROM ubuntu:20.04

WORKDIR /home

RUN apt-get update && apt-get install -y sudo git wget

COPY requirements.apt /home
RUN bash requirements.apt

COPY ./cldrive /home/cldrive
WORKDIR /home/cldrive
RUN bazel build -c opt //gpu/cldrive

ENTRYPOINT ["/bin/bash"]
