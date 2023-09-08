CUDA_VISIBLE_DEVICES=3 python3 main_for_search_negs.py --model=TransCF --data_path=data/ml-20m --lr=5e-3 --margin=1.0 --epoch=100 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 main_for_search_negs.py --model=TransCF --data_path=data/ml-20m --lr=5e-3 --margin=2.0 --epoch=100 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 main_for_search_negs.py --model=TransCF --data_path=data/ml-20m --lr=5e-3 --margin=0.5 --epoch=100 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 main_for_search_negs.py --model=TransCF --data_path=data/ml-20m --lr=5e-3 --margin=1.5 --epoch=100 --topks=3