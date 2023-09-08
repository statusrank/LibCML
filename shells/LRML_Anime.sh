CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=1.0 --dim=256 
CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=2.0 --dim=256 
CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=1.5 --dim=256 
CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=2.0 --dim=128
CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=1.5 --dim=128
CUDA_VISIBLE_DEVICES=1 python3 lrml_for_search_negs.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=1.0 --dim=128

CUDA_VISIBLE_DEVICES=1 python3 train_lrml.py --model=LRML --data_path=data/Anime --lr=1e-3 --margin=1.0 --dim=256 
