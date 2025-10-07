#!/bin/bash
set -e # 当任何命令失败时，立即退出脚本

# --- 1. 配置区 ---
echo "--- CONFIGURATION ---"
# 设置环境变量和GPU数量
NUM_GPUS=8 # 您服务器上的GPU数量
# 定义项目和文件的路径
ORTHUS_DIR="../"
BENCHMARK_DIR="../../datasets/SpatialViz" # 这里现在指向整个benchmark目录
# 定义临时和最终文件的名称
PREPROCESSED_FILE="orthus_input_for_svb.jsonl"
FINAL_INFERENCE_OUTPUT_FILE="orthus_lora_5epoch_instruct-question_text-only_output.jsonl"
CHUNK_DIR="temp_chunks" # 存放数据分片的临时目录
PARTIAL_OUTPUT_DIR="temp_lora_5epoch_partial_outputs" # 存放送部分结果的临时目录
# MODEL_CHECKPOINT="SJTU-Deng-Lab/Orthus-7B-instruct"
MODEL_CHECKPOINT="/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-loraV2"
BASE_MODEL="SJTU-Deng-Lab/Orthus-7B-instruct"
LOG_DIR="logs_lora" # <-- 新增：日志目录

echo "Model: ${MODEL_CHECKPOINT}"
echo "Benchmark Directory: ${BENCHMARK_DIR}"
echo "Parallel GPUs: ${NUM_GPUS}"
echo "---------------------"
echo

# --- 2. 预处理数据 (此步骤不变，只需执行一次) ---
echo "--- [STEP 1/5] Preprocessing benchmark data for Orthus... ---"
python preprocess_benchmark.py \
    --benchmark_dir "$BENCHMARK_DIR" \
    --output_file "$PREPROCESSED_FILE"
echo "Preprocessing successful."
echo

# --- 3. 将数据分片成8份 ---
echo "--- [STEP 2/5] Splitting input data into ${NUM_GPUS} chunks... ---"
# 清理并创建临时目录
rm -rf "$CHUNK_DIR"
mkdir -p "$CHUNK_DIR"

# 计算总行数和每个分片应有的行数（向上取整）
TOTAL_LINES=$(wc -l < "$PREPROCESSED_FILE")
LINES_PER_CHUNK=$(( (TOTAL_LINES + NUM_GPUS - 1) / NUM_GPUS ))

# 使用 split 命令进行分片
split -l "$LINES_PER_CHUNK" "$PREPROCESSED_FILE" "${CHUNK_DIR}/chunk_"
echo "Data split complete."
echo

# --- 4. 并行运行推理 (核心修改) ---
echo "--- [STEP 3/5] Starting parallel inference on ${NUM_GPUS} GPUs... ---"
rm -rf "$PARTIAL_OUTPUT_DIR" && mkdir -p "$PARTIAL_OUTPUT_DIR"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR" # <-- 新增：创建日志目录

# # 进入Orthus目录
# cd "$ORTHUS_DIR"

# 循环启动8个后台进程
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    CHUNK_SUFFIX=$(printf "%b" "$(printf '\\%o' $((i/26+97)))$(printf '\\%o' $((i%26+97)))")
    INPUT_CHUNK_FILE="./${CHUNK_DIR}/chunk_${CHUNK_SUFFIX}"
    OUTPUT_PARTIAL_FILE="./${PARTIAL_OUTPUT_DIR}/output_${i}.jsonl"
    LOG_FILE="./${LOG_DIR}/inference_gpu_${i}.log" # <-- 新增：定义每个进程的日志文件
    
    echo "Starting process for GPU ${i}. Progress logged to ${LOG_FILE}"

    # 【核心修改】使用 `2>` 将tqdm的进度条输出重定向到日志文件
    CUDA_VISIBLE_DEVICES=$i python ../inference/interleave_generation_lora.py \
        --input_file "$INPUT_CHUNK_FILE" \
        --base_model_path "$BASE_MODEL" \
        --lora_adapter_path "$MODEL_CHECKPOINT" \
        --output_file "$OUTPUT_PARTIAL_FILE" 2> "$LOG_FILE" &
done

echo "All inference processes started. Waiting for completion..."
wait
echo "All inference processes have finished."
echo

# --- 5. 合并结果 ---
echo "--- [STEP 4/5] Merging partial results... ---"
cat "${PARTIAL_OUTPUT_DIR}"/output_*.jsonl > "$FINAL_INFERENCE_OUTPUT_FILE"
echo "Merging complete. Final results are in ${FINAL_INFERENCE_OUTPUT_FILE}"
echo

# --- 6. 运行评估 ---
echo "--- [STEP 5/5] Evaluating the final results... ---"
python final_evaluate.py \
    --prediction_file "$FINAL_INFERENCE_OUTPUT_FILE" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --output_results_file "final_evaluation_results.json" # <-- 新增：指定输出文件
echo

# # --- 7. 清理临时文件 (可选) ---
# echo "--- Cleaning up temporary files... ---"
# rm -rf "$CHUNK_DIR" 
# echo

echo "--- Workflow Finished ---"