#!/bin/bash

#SBATCH -J "deep-learning-training"  # İşin adı
#SBATCH -A ***                    # Hesap adı
#SBATCH -p a100x4q                    # Partition (Kuyruk)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4                # 4 GPU istiyoruz
#SBATCH --time=2-00:00:00           # Maksimum süre
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kahramano20@itu.edu.tr 

# Hata olursa scripti anında durdur (Böylece boşuna devam etmez)
set -e

################################################################################
# Ayarlar
################################################################################
PROJECT_DIR=$(pwd)          # Scriptin çalıştırıldığı yeri proje dizini yap
VENV_PATH="./venv"          # Sanal ortam klasörü


# WandB ayarları
export WANDB_API_KEY="***"
export WANDB_MODE=online

################################################################################
# Başlangıç Bilgileri
################################################################################
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $PROJECT_DIR"
echo "Date: $(date)"

# Klasörleri oluştur
mkdir -p logs checkpoints

################################################################################
# 1. Modülleri Yükle
################################################################################
echo ""
echo "Loading modules..."

# Önceki modülleri temizle
module purge

# İTÜ sunucusu için tahmin edilen modüller (Hata alırsan versiyonları silip dene)
module load Python/Python-3.11.6-openmpi-4.1.6-gcc-11.4.0 || echo "Uyarı: Python modülü yüklenemedi, sistem varsayılanı kullanılacak."
module load cuda/cuda-12.5-a100q || echo "Uyarı: CUDA modülü yüklenemedi."

# ################################################################################
# # 2. Sanal Ortam Kurulumu ve Aktivasyonu (OTOMATİK)
# ################################################################################
# echo ""
# if [ ! -d "$VENV_PATH" ]; then
#     echo "Sanal ortam bulunamadı, oluşturuluyor: $VENV_PATH"
#     python -m venv "$VENV_PATH"
# else
#     echo "Sanal ortam mevcut, aktif ediliyor..."
# fi

# source "$VENV_PATH/bin/activate"

# ################################################################################
# # 3. Kütüphaneleri Yükle (requirements.txt) - BURASI EKLENDİ
# ################################################################################
# echo ""
# echo "Kütüphane kontrolü yapılıyor..."

# # pip'i güncelle (isteğe bağlı ama önerilir)
# pip install --upgrade pip

# if [ -f "requirements.txt" ]; then
#     echo "requirements.txt bulundu, eksik paketler yükleniyor..."
#     # Bu işlem Compute Node üzerinde çalıştığı için hata vermeyecek
#     pip install -r requirements.txt
# else
#     echo "UYARI: requirements.txt bulunamadı! Kurulum atlanıyor."
# fi

# # Ortam kontrolü
# echo ""
# echo "Ortam Kontrolü:"
# python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, GPU Count: {torch.cuda.device_count()}')"

# src klasörünü Python yoluna ekle
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

################################################################################
# 4. Eğitimi Başlat
################################################################################
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="

# Python scriptini çalıştır (-u: anlık log basması için)
torchrun --nproc_per_node=4 train.py

EXIT_CODE=$?

echo ""
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE