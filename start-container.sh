#!/bin/bash

CONTAINER_NAME="gym-city-container"

# Check if a container with the same name exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the new container
docker run --name $CONTAINER_NAME -d your-image-name
