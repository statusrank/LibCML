CUDA_VISIBLE_DEVICES=1 python3 train_SFCML.py --data_path=data/ml-100k/ \
                --lr=5e-2 \
                --margin=2.0 \
                --dim=256 \
                --epoch=300
                
CUDA_VISIBLE_DEVICES=2 python3 train_SFCML.py --data_path=data/Steam-200k/ \
                --lr=5e-2 \
                --margin=1.5 \
                --dim=256 \
                --epoch=300
