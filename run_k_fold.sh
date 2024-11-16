#!/bin/bash


#SBATCH --job-name=K-Fold-Q1
#SBATCH --nodes=1
#SBATCH --nodelist=hpc24
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20010736@st.phenikaa-uni.edu.vn


pip install -r requirements.txt

python test_k_fold.py --model_type adaptive_lstm --data_type static --scenario person_divide --num_classes 12 --epochs 300 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn ce  --normalizer batch_norm --seed 71 --k_fold 5
