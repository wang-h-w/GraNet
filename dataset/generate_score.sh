export TORCH_CUDA_ARCH_LIST="8.0"  # downgrade RTX3080
CUDA_VISIBLE_DEVICES=0 python generate_objectness.py --dataset_root DATASET_PATH --points 20000 --start 0 --end 190 --camera 'realsense'