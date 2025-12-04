while getopts c:i:b:p: flag
do
    case "${flag}" in
        c) COMPUTE=${OPTARG};;
        i) GPU_ID=${OPTARG};;
        b) BUILD=${OPTARG};;
        p) PORT=${OPTARG}
    esac
done

echo "COMPUTE: ${COMPUTE}";
echo "GPU_ID: ${GPU_ID}";
echo "PORT: ${PORT}";

if [ "${BUILD}" = "true" ];
then
    sudo docker build -f Dockerfiles/Dockerfile \
    --build-arg COMPUTE="${COMPUTE}" \
    --build-arg PORT="${PORT}" \
    --target vscode \
    -t mr-feature-extraction:vscode .
else
    echo "Use existing mr-feature-extraction:vscode"
fi

sudo docker run -itd --shm-size=100gb --gpus device="${GPU_ID}" -p ${PORT}:${PORT} \
-v "$(pwd)"/data:/mr-feature-extraction/data \
-v "$(pwd)"/src:/mr-feature-extraction/src \
-v "$(pwd)"/weights:/mr-feature-extraction/weights \
--name "mr-vscode-gpu-${GPU_ID}" mr-feature-extraction:vscode
