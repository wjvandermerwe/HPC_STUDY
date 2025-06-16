# CUDA toolkit + compiler
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# ----- build-time deps -----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        freeglut3-dev \
        libgl1-mesa-dev \
        libx11-dev && \
    rm -rf /var/lib/apt/lists/*

# ----- copy source -----
WORKDIR /app
#COPY . .
COPY Common ./Common
#COPY ./common ./common
COPY Lec6_codes/julia_cpu.cu .
# ----- build -----
# adjust include path if your code lives elsewhere
RUN nvcc -I./Common julia_cpu.cu -o julia_cpu \
        -lglut -lGL -lX11


# ----- default run -----
CMD ["./julia_cpu"]
