FROM nvidia/cudagl:10.0-base-ubuntu18.04
USER root
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip \
    libgtk-3-dev \
    gir1.2-gtk-3.0 \
	libgirepository1.0-dev \
    python3-gi \
    libgdk-pixbuf2.0-0 \
    libcairo2-dev \
    swig3.0 \
    python3-cairo-dev \
    git \
    libxdamage1 \
    libxfixes3

# This is weirdly necessary to fix broken `.so` files... (?)
RUN apt install --reinstall -y \
    libxdamage1 \
    libxfixes3

RUN pip3 install \
    pycairo \
    gym

# There seems to be a system-wide install already. But without this additional installation, we get a segfault.
RUN pip3 install --upgrade --force-reinstall \
    pygobject

# Copy the current directory contents into the container at /usr/src/app
WORKDIR /usr/src/app
COPY . ./
RUN make clean
RUN mkdir -p gym_city/envs/micropolis/MicropolisCore/src/TileEngine/objs
RUN mkdir -p gym_city/envs/micropolis/MicropolisCore/src/CellEngine/objs
RUN mkdir -p gym_city/envs/micropolis/MicropolisCore/src/MicropolisEngine/objs
RUN make
RUN make install

# This throws an error---I think because there is no way to render, neither from here nor an interactive shell.
# Instead, launch an interactive shell, then connect to the running container via the VSCode Dev Container extension,
# *then* run this command. This way, the GUI is able to render.
CMD python3 tilemap_test.py


# RUN pip3 install \
#   matplotlib \
# 	imutils \
# 	graphviz \
# 	visdom \
#   tensorflow \
#   torchsummary

#FROM anibali/pytorch:cuda-10.0
#FROM ubuntu:latest

# RUN apt-get install -y \
# 		python3-dev \
#         python3-gi \
#         python3-gi-cairo \
# 		python3-pip \
# 		libcairo2-dev \
# 		libgirepository1.0-dev \
# 		gcc \
# 		pkg-config \
# 		python3-dev \
# 		gir1.2-gtk-3.0  \
#         libopenmpi-dev \
#         python3-mpi4py \
#         git \
#         swig3.0 \
#         python3-cairo-dev \
#         libcanberra-gtk3-module \
#         libsm6 


# RUN pip3 install torch torchvision

# #WORKDIR /usr/src/app

# RUN git clone https://github.com/openai/baselines
# WORKDIR ./baselines
# RUN pip3 install -e .

# RUN mkdir trained_models; 
# COPY setup.py README.md ./
# RUN pip3 install -e .
# #COPY algo game_of_life ./
# #COPY *.py ./

# #RUN export GDK_SUNCHRONIZE=1
# #RUN export NO_AT_BRIDGE=1
# #RUN dbus-uuidgen > /var/lib/dbus/machine-id

# CMD python3 main.py \
#     --experiment DockerTest \ 
#     --render \
#     --overwrite
