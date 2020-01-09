FROM nvidia/cuda:10.0-base-ubuntu18.04
USER root
RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN pip3 install torch torchvision
CMD python3
#FROM anibali/pytorch:cuda-10.0
#FROM ubuntu:latest

RUN apt-get install -y python3-gi \
		python3-gi-cairo \
		gir1.2-gtk-3.0 \
		pkg-config \
		python3-pip

RUN apt install -y  \
		python3-dev \
		python3-pip \
		libcairo2-dev \
		libgirepository1.0-dev \
		gcc \
		pkg-config \
		python3-dev \
		gir1.2-gtk-3.0 

RUN pip3 install pycairo \
		PyGObject

#WORKDIR /usr/src/app
COPY setup.py README.md ./
RUN pip3 install -e .
RUN apt install -y python3-mpi4py
RUN apt install -y git
RUN git clone https://github.com/openai/baselines
WORKDIR ./baselines
RUN pip3 install tensorflow
RUN pip3 install -e .
RUN apt install -y libsm6
RUN pip3 install torchsummary \
		matplotlib \
		imutils \
		graphviz \
		visdom
WORKDIR /usr/src/app
RUN apt install -y swig3.0 python3-cairo-dev libcanberra-gtk3-module
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/TileEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/CellEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/MicropolisEngine/objs; exit 0

COPY . ./
RUN make ; exit 0
RUN make install; exit 0
CMD python3 -c 'import torch; print(torch.cuda.is_available())'
CMD python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'
#RUN export GDK_SUNCHRONIZE=1
CMD mkdir trained_models; exit 0
CMD python3 main.py --experiment DockerTest --render
