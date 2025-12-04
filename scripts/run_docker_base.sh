
while getopts c:i:b: flag
do
    case "${flag}" in
        c) COMPUTE=${OPTARG};;
        i) GPU_ID=${OPTARG};;
        b) BUILD=${OPTARG}
    esac
done

echo "COMPUTE: ${COMPUTE}";
echo "GPU_ID: ${GPU_ID}";

if [ "${BUILD}" = "true" ];
then
    docker build -f Dockerfiles/Dockerfile \
    --build-arg COMPUTE="${COMPUTE}" \
    --target base \
    -t mr-train-repo:base .
else
    echo "Use existing mr-train-repo:base"
fi

docker run -itd --shm-size=30gb --gpus device="${GPU_ID}" \
-v /home/jovyan/mr-data/:/mr-train-repo/data \
-v /home/jovyan/mr-train-repo/src:/mr-train-repo/src \
-v /home/jovyan/mr-train-repo/configs:/mr-train-repo/configs \
--name "mr-train-gpu-${GPU_ID}" mr-train-repo:base
