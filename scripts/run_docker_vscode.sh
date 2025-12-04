docker build --shm-size=50GB -t moment_detector:latest --target vscode -f Dockerfiles/Dockerfile .

sudo docker run --gpus 0 --name moments_trainer -itd -p 6001:6001 \
    -v /data/dataset/moment_retrieval/qvhighlights:/mr-train-repo/data \
    -v /data/reps/moment_retrieval/mr-train-repo/src:/mr-train-repo/src \
    -v /data/reps/moment_retrieval/mr-train-repo/configs:/mr-train-repo/configs \
    moment_detector:latest
