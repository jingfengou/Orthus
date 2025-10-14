export CUDA_VISIBLE_DEVICES=4,5,6,7

python decode_latents.py \
    --model_path "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v5" \
    --latents_file "debug_outputs/step_19_latents.pt"

# * `--model_path`: 指向您**训练好的模型**的文件夹（它需要模型里的VQVAE解码器）。
# * `--latents_file`: 指向您想要解码的那个 `.pt` 文件。
