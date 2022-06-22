
# Finetune
CUDA_VISIBLE_DEVICES=0 python main.py --dataset TAU-ASC --model_name BC-ResNet --mode finetune --data_root &
CUDA_VISIBLE_DEVICES=1 python main.py --dataset TAU-ASC --model_name BC-ResNet --mode finetune --data_root &
# Sampling
CUDA_VISIBLE_DEVICES=2 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage equal --data_root
CUDA_VISIBLE_DEVICES=0 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage random --data_root &
CUDA_VISIBLE_DEVICES=1 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage reservoir --data_root &
CUDA_VISIBLE_DEVICES=2 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage prototype --data_root
CUDA_VISIBLE_DEVICES=0 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage uncertainty --data_root &
# Uncertainty_Metric
CUDA_VISIBLE_DEVICES=1 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage uncertainty --uncert_metric shift--data_root &
CUDA_VISIBLE_DEVICES=2 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage uncertainty --uncert_metric noise --data_root
CUDA_VISIBLE_DEVICES=0 python main.py --dataset TAU-ASC --model_name BC-ResNet --mem_manage uncertainty --uncert_metric mask --data_root






