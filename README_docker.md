# Docker usage
Build the docker images with:
```
build . -t pywave
```
Build container with (eg.):
```
docker container rm -f pywave
docker create -it \
    -p 8889:8889 \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/pywave \
    --security-opt seccomp=unconfined \
    --cpus 4 \
    --name=pywave pywave
```
Can use different ports (eg `9999:9999` or `8888:8888`) if `8889` is taken.

Start container with:
```
docker start -i pywave
```
Launch notebook with:
```
jupyter-notebook --ip=0.0.0.0 --allow-root --port 8889 &
```
and paste the last URL printed into a web browser. Remember to use a different port if not using `8889`.
