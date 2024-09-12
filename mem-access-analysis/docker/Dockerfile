FROM ubuntu:22.04

# Update the package repository and install required packages
RUN apt-get update && apt-get install -y sudo git wget

# Clone the LLVM repository
WORKDIR /home/clang-llvm
RUN git clone https://github.com/llvm/llvm-project.git

# install dependencies
WORKDIR /home/
COPY requirements.apt .
RUN bash requirements.apt
## install python 3.8.15
SHELL ["/bin/bash", "-c"]
ENV HOME /home
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install 3.8.15
RUN pyenv global 3.8.15

# Build LLVM and Clang
WORKDIR /home/clang-llvm/build
RUN python3 --version
RUN cmake -G Ninja ../llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_TESTS=ON  # Enable tests; default is off.
## build clang and llvm
RUN ninja
RUN ninja install
## setting clang as default compiler
RUN apt-get install -y cmake-curses-gui libedit-dev ocaml vim \
    && ln -s /home/clang-llvm/build/bin/clang++ /usr/bin/clang++
RUN cmake -S ../llvm-project/llvm -B build -DCMAKE_CXX_COMPILER="/usr/bin/clang++" -DCMAKE_BUILD_TYPE=Release
RUN ninja


WORKDIR /home
CMD ["/bin/bash"]
