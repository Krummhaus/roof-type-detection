# Get your host UID and GID
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Build docker image
docker build --build-arg UID=$HOST_UID --build-arg GID=$HOST_GID -t bp_pytorch .
