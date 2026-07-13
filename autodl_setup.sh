#!/bin/bash
# AutoDL 部署脚本
# 数据集 CSVs 已包含在仓库中，无需重复 01-04 预处理步骤。
# 在新 GPU 主机上运行：
#   bash autodl_setup.sh

set -e

echo "=== 1. 拉取最新代码 ==="
cd /root
if [ -d horti-m3-large-dafn ]; then
    cd horti-m3-large-dafn && git pull
else
    git clone https://github.com/braverain08/horti-m3-large-dafn.git
    cd horti-m3-large-dafn
fi

echo "=== 2. 安装依赖 ==="
pip install -r requirements_autodl.txt

echo "=== 3. 解压数据集 ==="
DATASET_ZIP="/root/2023-2025 Tomato dataset.zip"
if [ ! -d "/root/2023-2025 Tomato dataset" ]; then
    if [ -f "$DATASET_ZIP" ]; then
        unzip "$DATASET_ZIP" -d /root/
    else
        echo "下载数据集 (~5GB)..."
        wget -O "$DATASET_ZIP" \
          'https://zenodo.org/records/17217565/files/2023-2025%20Tomato%20dataset.zip?download=1'
        unzip "$DATASET_ZIP" -d /root/
    fi
fi

echo "=== 4. 提取图像特征 (GPU) ==="
python data/05_extract_features.py \
    --input data/dataset_ready.csv \
    --output data/image_features.npy \
    --data-dir "/root/2023-2025 Tomato dataset/" \
    --batch_size 128

echo "=== 5. 提取微调特征 (GPU) ==="
python data/06_finetune_resnet.py \
    --csv data/dataset_ready.csv \
    --data-dir "/root/2023-2025 Tomato dataset/" \
    --output data/finetuned_logits.npy \
    --epochs 10

echo "=== 完成 ==="
echo "生成的 .npy 文件在 data/ 目录，请下载回本地:"
echo "  scp root@<autoDL-IP>:~/horti-m3-large-dafn/data/image_features.npy ./data/"
echo "  scp root@<autoDL-IP>:~/horti-m3-large-dafn/data/finetuned_logits.npy ./data/"
