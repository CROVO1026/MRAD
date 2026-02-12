#!/bin/bash

# ============================================
# MRAD Training and Testing Pipeline
# Train on VISA, test on MVTEC
# ============================================

# Environment setup
# Python 3.9, PyTorch 2.0.0+cu117
# pip install -r requirements.txt

# Model type parameter (options: mrad-clip, mrad-ft)
MODEL_TYPE=${1:-"mrad-clip"}

# Get project directory (current script location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path configuration
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/released_${MODEL_TYPE}"
LOG_DIR="${PROJECT_DIR}/logs/released_${MODEL_TYPE}"
CACHE_DIR="${PROJECT_DIR}/cache"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"
TEST_SCRIPT="${PROJECT_DIR}/test.py"

# Dataset configuration (users should modify these paths)
TRAIN_DATASET="visa"
TRAIN_DATA_PATH="${PROJECT_DIR}/data/VISA"
TEST_DATASET="mvtec"
TEST_DATA_PATH="${PROJECT_DIR}/data/MVTEC"

# Training GPU
TRAIN_GPU="cuda:1"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "MRAD Pipeline"
echo "Model Type: ${MODEL_TYPE}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Test Dataset: ${TEST_DATASET}"
echo "============================================"

# ============================================
# Stage 1: Training
# ============================================
echo ""
echo "[Stage 1] Training on ${TRAIN_DATASET}..."
TRAIN_LOG_FILE="${LOG_DIR}/train.log"

nohup python $TRAIN_SCRIPT \
    --model_type "$MODEL_TYPE" \
    --dataset "$TRAIN_DATASET" \
    --data_path "$TRAIN_DATA_PATH" \
    --save_path "$CHECKPOINT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --device "$TRAIN_GPU" \
    > "$TRAIN_LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo "Waiting for training to complete..."
wait $TRAIN_PID
echo "[Stage 1] Training completed. Log: $TRAIN_LOG_FILE"

# ============================================
# Stage 2: Testing
# ============================================
echo ""
echo "[Stage 2] Testing on ${TEST_DATASET}..."

if [ "$MODEL_TYPE" == "mrad-ft" ]; then
    # MRAD-FT trains only 1 epoch, test epoch 1 only
    echo "Testing MRAD-FT (epoch 1 only)..."

    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mrad_ft_epoch_1.pth"
    LOG_FILE="${LOG_DIR}/test_epoch_1.log"

    python $TEST_SCRIPT \
        --model_type "mrad-ft" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --dataset "$TEST_DATASET" \
        --data_path "$TEST_DATA_PATH" \
        --cache_dir "$CACHE_DIR" \
        --save_path "${LOG_DIR}/results" \
        --device "cuda:0" \
        2>&1 | tee "$LOG_FILE"

    echo "MRAD-FT testing completed."

else
    # MRAD-CLIP trains 6 epochs, test all epochs (distributed across multiple GPUs)
    echo "Testing MRAD-CLIP (epochs 1-6 in parallel)..."

    # cuda:0: epoch 1, 5
    for epoch in 1 5; do
        GPU_ID="cuda:0"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}/mrad_clip_epoch_${epoch}.pth"
        LOG_FILE="${LOG_DIR}/test_epoch_${epoch}.log"

        if [ $epoch -eq 1 ]; then
            echo "Running test for epoch ${epoch} on ${GPU_ID} (mrad-ft mode)..."
            nohup python $TEST_SCRIPT \
                --model_type "mrad-ft" \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --dataset "$TEST_DATASET" \
                --data_path "$TEST_DATA_PATH" \
                --cache_dir "$CACHE_DIR" \
                --save_path "${LOG_DIR}/results_epoch_${epoch}" \
                --device "$GPU_ID" \
                > "$LOG_FILE" 2>&1 &
        else
            echo "Running test for epoch ${epoch} on ${GPU_ID} (mrad-clip mode)..."
            nohup python $TEST_SCRIPT \
                --model_type "mrad-clip" \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --dataset "$TEST_DATASET" \
                --data_path "$TEST_DATA_PATH" \
                --cache_dir "$CACHE_DIR" \
                --save_path "${LOG_DIR}/results_epoch_${epoch}" \
                --device "$GPU_ID" \
                > "$LOG_FILE" 2>&1 &
        fi
    done

    # cuda:1: epoch 2, 6
    for epoch in 2 6; do
        GPU_ID="cuda:1"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}/mrad_clip_epoch_${epoch}.pth"
        LOG_FILE="${LOG_DIR}/test_epoch_${epoch}.log"

        echo "Running test for epoch ${epoch} on ${GPU_ID} (mrad-clip mode)..."
        nohup python $TEST_SCRIPT \
            --model_type "mrad-clip" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --dataset "$TEST_DATASET" \
            --data_path "$TEST_DATA_PATH" \
            --cache_dir "$CACHE_DIR" \
            --save_path "${LOG_DIR}/results_epoch_${epoch}" \
            --device "$GPU_ID" \
            > "$LOG_FILE" 2>&1 &
    done

    # cuda:2: epoch 3
    for epoch in 3; do
        GPU_ID="cuda:2"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}/mrad_clip_epoch_${epoch}.pth"
        LOG_FILE="${LOG_DIR}/test_epoch_${epoch}.log"

        echo "Running test for epoch ${epoch} on ${GPU_ID} (mrad-clip mode)..."
        nohup python $TEST_SCRIPT \
            --model_type "mrad-clip" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --dataset "$TEST_DATASET" \
            --data_path "$TEST_DATA_PATH" \
            --cache_dir "$CACHE_DIR" \
            --save_path "${LOG_DIR}/results_epoch_${epoch}" \
            --device "$GPU_ID" \
            > "$LOG_FILE" 2>&1 &
    done

    # cuda:3: epoch 4
    for epoch in 4; do
        GPU_ID="cuda:3"
        CHECKPOINT_PATH="${CHECKPOINT_DIR}/mrad_clip_epoch_${epoch}.pth"
        LOG_FILE="${LOG_DIR}/test_epoch_${epoch}.log"

        echo "Running test for epoch ${epoch} on ${GPU_ID} (mrad-clip mode)..."
        nohup python $TEST_SCRIPT \
            --model_type "mrad-clip" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --dataset "$TEST_DATASET" \
            --data_path "$TEST_DATA_PATH" \
            --cache_dir "$CACHE_DIR" \
            --save_path "${LOG_DIR}/results_epoch_${epoch}" \
            --device "$GPU_ID" \
            > "$LOG_FILE" 2>&1 &
    done

    echo "All tests launched in background."
fi

echo ""
echo "============================================"
echo "Pipeline completed!"
echo "Logs saved to: ${LOG_DIR}/"
echo "============================================"

# View processes: ps -ef | grep test.py
# Kill processes: pkill -f "python.*test.py"
echo "============================================"
echo "Pipeline completed!"
echo "Logs saved to: ${LOG_DIR}/"
echo "============================================"

# 查看进程: ps -ef | grep test_released.py
# 终止进程: pkill -f "python.*test_released.py"
