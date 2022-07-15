# DCASE 2019 Task 1

# Finetune
python main.py --dataset TAU-ASC --mode finetune

# Random
python main.py --dataset TAU-ASC --mode replay --mem_manage random

# Reservoir
python main.py --dataset TAU-ASC --mode replay --mem_manage reservoir

# Prototype
python main.py --dataset TAU-ASC --mode replay --mem_manage prototype

# Uncertainty shift
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric shift --metric_k 2
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric shift --metric_k 4
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric shift --metric_k 6

# Uncertainty noise
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noise --metric_k 2
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noise --metric_k 4
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noise --metric_k 6

# Uncertainty++
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noisytune --metric_k 2
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noisytune --metric_k 4
python main.py --dataset TAU-ASC --mode replay --mem_manage uncertainty --uncert_metric noisytune --metric_k 6

