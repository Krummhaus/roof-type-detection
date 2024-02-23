docker run \
	-it \
	--gpus all \
	-v $(pwd):/home/nonroot \
	--user $(id -u):$(id -g) \
	-d bp_pytorch
