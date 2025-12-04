<div align="center"><h2>Documentation</h2></div>

This repository contains feature extractors for the moment retrieval task.

### 1. Build Docker

To **build and run the Docker container**, use the script [run_docker.sh](scripts/run_docker_base.sh).

**Note:** If you're building Docker for a GPU other than the V100, you need to set the `COMPUTE` value in [run_docker.sh#L4](scripts/run_docker_base.sh#L4). <br>
You can find the compute capability of your GPU [here](https://developer.nvidia.com/cuda-gpus). For example, the compute capability for V100 is 70.

---

### 2. Run Feature Extraction

#### Download Model Weights

To run feature extraction, download model weights from [this link](https://drive.google.com/drive/folders/1YnKJV0vju1Hfx6l2b4rraeTLhRY7uJp9?usp=sharing).

#### Video Feature Extraction

To extract video features, run the following command:

```bash
python src/cli/extract_features.py \
    --video_folder "your/path/to/videos" \
    --batch_size 32 \
    --batch_interval 600 \
    --sample_rate 16000 \
    --interval_duration 2 \
    --num_workers 7 \
    --video_checkpoint "path/to/video/weights/video_encoder.pt" \
    --audio_checkpoint "path/to/audio/weights/encoder.pth" \
    --raw_audio_checkpoint "path/to/raw_audio/weights/*.pt" \
    --output_folder "path/to/output/folder" \
    --load_to_s3 False
```

#### Text Feature Extraction

To extract text features, use this command:

```bash
python src/cli/extract_text_features.py \
    --captions_path "path/to/your/captions.csv" \
    --checkpoint "path/to/video/text_encoder.pt" \
    --output_folder "path/to/output/folder"
```
