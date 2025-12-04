#!/bin/bash

ulimit -n 128000
export PYTHONPATH=$(pwd)
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

# Getting the log_dir and exp_list_name values from Hydra
log_dir=$(python -c "from hydra import initialize, compose; initialize(config_path='./configs', version_base=None); cfg = compose(config_name='finetune_youtube'); print(cfg.paths.log_dir)")
exp_list_name=$(python -c "from hydra import initialize, compose; initialize(config_path='./configs', version_base=None); cfg = compose(config_name='finetune_youtube'); print(cfg.exp_list_name)")

# Loop over seed values
for seed in 45
do
    # Loop over domain values
    for domain in dog gymnastics parkour skating skiing surfing
    do
        # Construct the task_name as {seed}_{domain}
        task_name="${seed}_${domain}"

        # Define the output directory using exp_list_name, task_name, and current date-time
        output_dir="${log_dir}/${exp_list_name}/${task_name}/$(date +%Y-%m-%d-%H-%M-%S)"

        # Execute the command
        python ./src/cli/train.py --config-path /data/reps/mr-train-repo/configs/ \
        --config-name finetune_youtube data.dataset.domain=${domain} task_name=${task_name} seed=${seed} \
        hydra.run.dir=${output_dir}
    done
done

# collect metrics from different domains
python ./src/cli/collect_metrics.py --logs_directory "${log_dir}/${exp_list_name}" --metric_name "test/HL-mAP-Binary"
