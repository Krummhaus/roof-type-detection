FROM pytorch/pytorch:latest
ARG UID
ARG GID

# Install additional libraries
RUN addgroup --gid $GID nonroot && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    conda install -y pandas && \
    conda install conda-forge::matplotlib && \
    conda install -n base ipykernel --update-deps --force-reinstall -y && \
    conda install -c conda-forge --name base tensorboard -y

# Set the non-root user as the default user
USER nonroot

# Set the working directory
WORKDIR /home/nonroot

# Copy files into the container and set the appropriate permissions
#COPY --chown=nonroot:nonroot /home/nonroot/app /home/nonroot/app
RUN chmod -R 755 /home/nonroot
