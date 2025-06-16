docker build -t julia-cuda -f Lec6_codes/Dockerfile .
docker run --gpus all -it --rm  -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix  --entrypoint /bin/bash julia-cuda
docker cp stupefied_fermat:/app/julia.ppm .