# NPROC_PER_NODE 是你希望使用的 GPU 数量
export NPROC_PER_NODE=8

accelerate launch --num_processes ${NPROC_PER_NODE} train_orthus.py \
    --ckpt_path "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-think-v2" \
    --train_file "/data1/oujingfeng/project/twgi/datasets/SpatialViz/datasets/train.jsonl" \
    --eval_file "/data1/oujingfeng/project/twgi/datasets/SpatialViz/datasets/test.jsonl" \
    --image_folder "/data1/oujingfeng/project/twgi/datasets/SpatialViz" \
    --output_dir "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-think-v3" \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --early_stopping_patience 5 # <-- 在这里激活早停，耐心值为5
    # --generation_log_file "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-v4/generation_log.jsonl"