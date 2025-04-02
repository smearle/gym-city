# FROM nvidia/cuda:10.0-base-ubuntu18.04
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
USER root
RUN apt update
RUN apt upgrade -y
RUN apt install python3.12-venv -y
# RUN apt install -y python3-pip
RUN python3.12 -m venv venv
RUN /bin/bash -c "source venv/bin/activate && pip install torch torchvision"
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
RUN apt-get install -y cmake
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y girepository-2.0

RUN /bin/bash -c "source venv/bin/activate && pip install pycairo"
RUN /bin/bash -c "source venv/bin/activate && pip install PyGObject"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade pip"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade setuptools"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade wheel"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade numpy"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade gym"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade matplotlib"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade imutils"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade graphviz"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade visdom"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade tensorflow"
RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade torchsummary"

# RUN /bin/bash -c "source venv/bin/activate && python3.12 tilemap_test.py"

# RUN git clone https://github.com/openai/baselines
# WORKDIR ./baselines
# RUN /bin/bash -c "source ../venv/bin/activate && pip install -e ."

WORKDIR /usr/src/app
COPY . ./

RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/TileEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/CellEngine/objs; exit 0
RUN mkdir gym_city/envs/micropolis/MicropolisCore/src/MicropolisEngine/objs; exit 0
RUN make ; exit 0
RUN make install; exit 0
WORKDIR /usr/src/app/gym-city
CMD ls .
CMD /bin/bash -c "test -d /usr/src/app/venv && source /usr/src/app/venv/bin/activate && python3 /usr/src/app/tilemap_test.py"

# CMD python3 -c 'import torch; print(torch.cuda.is_available())'
# CMD python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())'
# RUN mkdir trained_models; 
# COPY setup.py README.md ./
# RUN /bin/bash -c "source venv/bin/activate && pip install -e ."
# #COPY algo game_of_life ./
# #COPY *.py ./

# #RUN export GDK_SUNCHRONIZE=1
# #RUN export NO_AT_BRIDGE=1
# #RUN dbus-uuidgen > /var/lib/dbus/machine-id

# CMD python3 main.py \
#     --experiment DockerTest \ 
#     --render \
#     --overwrite
