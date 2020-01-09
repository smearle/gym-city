FROM nvidia/cuda:10.0-base-ubuntu18.04
USER root
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip
RUN pip3 install torch torchvision
CMD python3
#FROM anibali/pytorch:cuda-10.0
#FROM ubuntu:latest

RUN apt-get install -y \
		python3-dev \
        python3-gi \
        python3-gi-cairo \
		python3-pip \
		libcairo2-dev \
		libgirepository1.0-dev \
		gcc \
		pkg-config \
		python3-dev \
		gir1.2-gtk-3.0  \
        libopenmpi-dev \
        python3-mpi4py \
        git \
        swig3.0 \
        python3-cairo-dev \
        libcanberra-gtk3-module \
        libsm6 

RUN pip3 install pycairo \
		PyGObject \
        matplotlib \
		imutils \
		graphviz \
		visdom \
        tensorflow \
        torchsummary

#WORKDIR /usr/src/app

RUN git clone https://github.com/openai/baselines
WORKDIR ./baselines
RUN pip3 install -e .

WORKDIR /usr/src/app
COPY . ./
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/TileEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/CellEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/MicropolisEngine/objs; exit 0
RUN make ; exit 0
RUN make install; exit 0
CMD python3 -c 'import torch; print(torch.cuda.is_available())'
CMD python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'
RUN mkdir trained_models; 
COPY setup.py README.md ./
RUN pip3 install -e .
#COPY algo game_of_life ./
#COPY *.py ./

#RUN export GDK_SUNCHRONIZE=1
#RUN export NO_AT_BRIDGE=1
#RUN dbus-uuidgen > /var/lib/dbus/machine-id

CMD python3 main.py \
    --experiment DockerTest \ 
    --render \
    --overwrite 
