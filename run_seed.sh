#!/bin/bash


#SBATCH --job-name=Q1_50
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

# MAX_SEED=50
# for ((i=1; i<=$MAX_SEED; i++))
# do
#     python main.py --model_type adaptive_lstm --data_type static --scenario person_divide --num_classes 12 --epochs 300 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn nll  --normalizer batch_norm --seed $i

# done

python main.py --model_type cnn_gru_bilstm --data_type static --scenario person_divide --num_classes 12 --epochs 2 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn nll  --normalizer batch_norm --seed 50

python main.py --model_type cnn_bilstm_gru --data_type static --scenario person_divide --num_classes 12 --epochs 2 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn nll  --normalizer batch_norm --seed 50

python main.py --model_type cnn_gru_att --data_type static --scenario person_divide --num_classes 12 --epochs 2 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn nll  --normalizer batch_norm --seed 50

python main.py --model_type cnn_gru_bilstm_att --data_type static --scenario person_divide --num_classes 12 --epochs 2 --sequence_length 100 --overlap 0.4 --batch_size 512 --loss_fn nll  --normalizer batch_norm --seed 50
