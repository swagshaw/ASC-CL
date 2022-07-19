[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)

# ASC-CL
Official Pytorch Implementation for [Continual Learning For On-Device Environmental Sound Classification](https://arxiv.org/abs/2207.07429)

If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/swagshaw/ASC-CL/issues/new) or [send me an email](mailto:yxiao009+github@e.ntu.edu.sg). 
## Abstract
Continuously learning new classes without catastrophic forgetting is a challenging problem for on-device environmental sound classification given the restrictions on computation resources (e.g., model size, running memory). To address this issue, we propose a simple and efficient continual learning method. Our method selects the historical data for the training by measuring the per-sample classification uncertainty. Specifically, we measure the uncertainty by observing how the classification probability of data fluctuates against the parallel perturbations added to the classifier embedding. In this way, the computation cost can be significantly reduced compared with adding perturbation to the raw data. Experimental results on the DCASE 2019 Task 1 and ESC-50 dataset show that our proposed method outperforms baseline continual learning methods on classification accuracy and computational efficiency, indicating our method can efficiently and incrementally learn new classes without the catastrophic forgetting problem for on-device environmental sound classification.
## Getting Started
### Setup Environment

You need to create the running environment by [Anaconda](https://www.anaconda.com/),

```bash
conda env create -f environment.yml
conda active asc
```
### Results
There are three types of logs during running experiments; logs, results. 
The log files are saved in `logs` directory, and the results which contains accuracy of each task and memory updating time are saved in `workspace` directory. 
```angular2html
workspace
    |_ logs 
        |_ [dataset]
            |_.log
            |_ ...
    |_ results
        |_ [dataset]
            |_.npy
            |_...
```
### Data

We use the [TAU-ASC](https://zenodo.org/record/2589280#.YtJiNHbP1UE) and [ESC-50](https://github.com/karoldvl/ESC-50/archive/master.zip) dataset as the training data.
You should put them into:

```bash
your_project_path/data/TAU_ASC
your_project_path/data/ESC-50-master
```

Then use the `./data/generate_json.py`:

```bath
python ./data/generate_json.py --mode train --dpath your_project_path /data
python ./data/generate_json.py --mode test --dpath your_project_path /data
```

### Usage

To run the experiments in the paper, you just run `train.sh`.
For various experiments, you should know the role of each argument.

- `MODE`: use CL method or not [finetune, replay]
- `MODEL`: use baseline CNN model or BC-ResNet  [baseline, BC-ResNet ]
- `MEM_MANAGE`: Memory update method.[random, reservoir, uncertainty, prototype].
- `RND_SEED`: Random seed number
- `DATASET`: Dataset name [TAU-ASC, ESC-50]
- `MEM_SIZE`: Memory size: k={300, 500}
- `UNCERT_MERTIC`: Perturbation methods for uncertainty [shift, noise, noisytune(ours)]

## Acknowledgements
Our implementations use the source code from the following repositories and users:

- [Rainbow-Keywords](https://github.com/swagshaw/Rainbow-Keywords)

## License
The project is available as open source under the terms of the [MIT License](./LICENSE).
