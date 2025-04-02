FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# Set working directory inside container
WORKDIR /usr/src/app

# Install dependencies
RUN apt update && apt upgrade -y && \
    apt-get install -y \
        python3.12-venv \
        python3-dev \
        python3-gi \
        python3-gi-cairo \
        python3-pip \
        libcairo2-dev \
        libgirepository1.0-dev \
        gcc \
        pkg-config \
        gir1.2-gtk-3.0 \
        libopenmpi-dev \
        python3-mpi4py \
        git \
        swig3.0 \
        python3-cairo-dev \
        libcanberra-gtk3-module \
        libsm6 \
        cmake \
        libgtk-3-dev \
        girepository-2.0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the gym-city repo into the container
COPY . ./gym-city

# Set the working directory to the gym-city repo
WORKDIR /usr/src/app/gym-city

# Create virtual environment inside gym-city
RUN python3.12 -m venv venv

# Activate venv and install Python dependencies
RUN /bin/bash -c "source venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision tensorflow gym numpy matplotlib imutils graphviz visdom torchsummary pycairo PyGObject"

# Create required build directories before make
RUN mkdir -p \
    envs/micropolis/MicropolisCore/src/TileEngine/objs \
    envs/micropolis/MicropolisCore/src/CellEngine/objs \
    envs/micropolis/MicropolisCore/src/MicropolisEngine/objs

# Build and install the Micropolis engine (ignore failure if `make install` is not present)
RUN make || true && make install || true

# Default command to run
CMD ["./venv/bin/python", "tilemap_test.py"]


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
