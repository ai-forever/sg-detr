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
    sudo docker build -f Dockerfiles/Dockerfile \
    --build-arg COMPUTE="${COMPUTE}" \
    --target base \
    -t mr-feature-extraction:base .
else
    echo "Use existing mr-feature-extraction:base"
fi

sudo docker run -itd --shm-size=100gb --gpus device="${GPU_ID}" \
-v "$(pwd)"/data:/mr-feature-extraction/data \
-v "$(pwd)"/weights:/mr-feature-extraction/weights \
--name "mr-pretrain_gpu_${GPU_ID}" mr-feature-extraction:base
