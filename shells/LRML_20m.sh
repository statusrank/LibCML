CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=1.0 --dim=256 --topks=3 
CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=2.0 --dim=256 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=1.5 --dim=256 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=2.0 --dim=128 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=1.5 --dim=128 --topks=3
CUDA_VISIBLE_DEVICES=3 python3 lrml_for_search_negs.py --model=LRML --data_path=data/ml-20m --lr=1e-3 --margin=1.0 --dim=128 --topks=3