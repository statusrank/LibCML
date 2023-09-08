# CUDA_VISIBLE_DEVICES=3 python3 train_CPE.py --model=CPE --data_path=data/ml-100k --lr=5e-2 --margin=1.2 --sampling_strategy='hard' --cov_loss_reg=5e-5
# CUDA_VISIBLE_DEVICES=2 python3 train_CPE.py --model=CPE --data_path=data/Steam-200k --lr=5e-2 --margin=2.0 --sampling_strategy='hard' --cov_loss_reg=0
# CUDA_VISIBLE_DEVICES=3 python3 train_CPE.py --model=CPE --data_path=data/ml-100k --lr=5e-2 --margin=2.0 



GPUS1=(0 1 2 3)
GPUS2=(4 5 6 7)
MARGINS=(0.5 0.8 1.0 1.2 1.5 2.0)
LRS=(0.01 0.03 0.05 0.005)

for idy in $(seq 0 5)
    do
    for idx in $(seq 0 3)
        do 
            
            CUDA_VISIBLE_DEVICES=${GPUS1[$idx]} python3 train_CPE.py --data_path=data/ml-100k/ \
                --lr=${LRS[$idx]} \
                --margin=${MARGINS[$idy]} \
                --model=CPE \
                --dim=256 \
                --sampling_strategy='hard' \
                --cov_loss_reg=1e-6 \
                --epoch=300 &
            
            CUDA_VISIBLE_DEVICES=${GPUS2[$idx]} python3 train_CPE.py --data_path=data/Steam-200k/ \
                --lr=${LRS[$idx]} \
                --margin=${MARGINS[$idy]} \
                --model=CPE \
                --dim=256 \
                --sampling_strategy='hard' \
                --cov_loss_reg=1e-6 \
                --epoch=300 &
        done
    wait
    done