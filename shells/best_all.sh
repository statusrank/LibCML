
# CUDA_VISIBLE_DEVICES=3 python3 train_example.py --model=CPE --data_path=data/ml-100k --lr=5e-2 --margin=1.2 --sampling_strategy='hard' --cov_loss_reg=0.0 --dim=256

# CUDA_VISIBLE_DEVICES=3 python3 train_example.py --model=SFCML --data_path=data/ml-100k --lr=5e-2 --margin=2.0 --sampling_strategy='hard' --dim=256

CUDA_VISIBLE_DEVICES=4 python3 train_example.py --model=LRML --data_path=data/ml-100k --lr=1e-2 --margin=1.5 --sampling_strategy='uniform' --dim=256 --num_mems=20 --epoch=300

CUDA_VISIBLE_DEVICES=4 python3 train_example.py --model=TransCF --data_path=data/ml-100k --lr=1e-3 --margin=2.0 --sampling_strategy='uniform' --dim=256 --epoch=300 --dis_reg=0.01 --nei_reg=0.01

CUDA_VISIBLE_DEVICES=5 python3 train_example.py --model=CRML --data_path=data/ml-100k --lr=5e-3 --margin=2.0 --sampling_strategy='uniform' --dim=256 --epoch=300 --alpha=0.1 --beta=0.1

CUDA_VISIBLE_DEVICES=0 python3 train_example.py \
    --data_path=data/ml-100k \
    --model=COCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=1 \
    --sampling_strategy=uniform \
    --dim=256 \
    --DCRS_reg=0.  \
    --epoch=300

CUDA_VISIBLE_DEVICES=1 python3 train_example.py \
    --data_path=data/ml-100k \
    --model=HarCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=1 \
    --sampling_strategy=hard \
    --dim=256 \
    --DCRS_reg=0.  \
    --epoch=300

CUDA_VISIBLE_DEVICES=0 python3 train_example.py \
    --data_path=data/ml-100k \
    --model=COCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=3 \
    --sampling_strategy=uniform \
    --dim=256 \
    --DCRS_reg=10  \
    --m1=0.05 \
    --m2=0.25 \
    --epoch=300

CUDA_VISIBLE_DEVICES=0 python3 train_example.py \
    --data_path=data/ml-100k \
    --model=HarCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=3 \
    --sampling_strategy=hard \
    --dim=256 \
    --DCRS_reg=10  \
    --m1=0.05 \
    --m2=0.25 \
    --epoch=300