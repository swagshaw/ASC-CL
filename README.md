# asc-continual-learning
Pytorch Implementation for Continual Learning of Acoustic Scene Classification (ASC)

### Setup Environment

You need to create the running environment by [Anaconda](https://www.anaconda.com/),

```bash
conda env create -f environment.yml
conda active asc
```
### Data
We use the TAU-ASC and ESC-50 dataset as the training data. 
You should put them into:
```bash
./data/TAU_ASC
./data/ESC-50-master
```
### Usage 
To run the experiments in the paper, you just run `train.sh`.
For various experiments, you should know the role of each argument. 

- `MODE`: use CL method or not [finetune, replay] 
- `MODEL`: use baseline CNN model or BC-ResNet  [baseline, BC-ResNet ]
- `MEM_MANAGE`: Memory management method.[equal, random, reservoir, uncertainty, prototype].
- `RND_SEED`: Random Seed Number 
- `DATASET`: Dataset name [TAU-ASC, ESC-50]
- `MEM_SIZE`: Memory size: k={300, 500, 1000, 1500}
- `UNCERT_MERTIC`: Metric for uncertainty [shift, noise, mask, combination]