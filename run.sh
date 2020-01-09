# This runs the Dockerfile, which has pytorch with gpu acceleration, and renders the environment.
# TODO: model-saving and logging
cp .gitignore .dockerignore
sudo xhost +local:root
sudo docker image build -t city:latest . && sudo docker run --gpus all --ipc=host --user="$(id -u):$(id -g)" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -e NVIDIA_VISIBLE_DEVICES=0 -e "DISPLAY" city:latest

