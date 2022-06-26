import argparse
import random
from collections import defaultdict
import torch
import os
import time
import numpy as np
import logging as log_config
from pytorch.losses import get_loss_func
from data_loader import get_train_datalist, get_test_datalist
from models.model import Baseline_CNN, BCResNet_Mod
from models.frontend import Audio_Frontend
from utils.get_methods import get_methods


def save_model(model, optimizer, step, acc, name):
    save_path = os.path.join(ckpt_dir, name + '.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'acc': acc
    }, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # Data root.
    parser.add_argument("--data_root", type=str, default='/home/xiaoyang/Dev/asc-continual-learning/data/collection')
    parser.add_argument('--exp_name', type=str, default='disjoint')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_name', type=str, default='BC-ResNet')  # 'baseline' | 'BC-ResNet'
    parser.add_argument('--dataset', type=str, default='ESC-50')  # 'TAU-ASC' | 'ESC-50' |
    parser.add_argument("--mode", type=str, default="finetune", help="CIL methods [finetune, replay]", )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default='prototype',
        help="memory management [random, uncertainty, reservoir, prototype]",
    )
    parser.add_argument("--n_tasks", type=int, default=11, help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=4, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=10,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, default=3, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="noisytune",
        choices=["shift", "noise", "mask", "combination", "noisytune"],
        help="A type of uncertainty metric",
    )
    parser.add_argument("--metric_k", type=int, default=4, choices=[2, 4, 6],
                        help="The number of the uncertainty metric functions")
    parser.add_argument("--noise_lambda", type=float, default=0.4,
                        help="The number of the uncertainty metric functions")
    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    args = parser.parse_args()
    if args.mode == "finetune":
        save_path = f"{args.dataset}_{args.mode}_cls{args.n_cls_a_task}" \
                    f"_epoch{args.epoch}_lr{args.lr}_rnd{args.rnd_seed}"
    elif args.mem_manage == "uncertainty":
        save_path = f"{args.dataset}_{args.mode}_cls{args.n_cls_a_task}_{args.mem_manage}_{args.uncert_metric}" \
                    f"_{args.metric_k}_{args.noise_lambda}_epoch{args.epoch}" \
                    f"_lr{args.lr}_msz{args.memory_size}_rnd{args.rnd_seed}"
    else:
        save_path = f"{args.dataset}_{args.mode}_cls{args.n_cls_a_task}_{args.mem_manage}" \
                    f"_epoch{args.epoch}_lr{args.lr}_msz{args.memory_size}_rnd{args.rnd_seed}"

    # Training parameters
    exp_name = args.exp_name
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.lr
    model_name = args.model_name
    dataset = args.dataset

    # Log file initalization
    ckpt_dir = os.path.join('workspace', dataset, exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = os.path.join(ckpt_dir, 'last.pt')
    ckpt_path = ckpt_name if os.path.exists(ckpt_name) else None

    log_dir = os.path.join('workspace', dataset, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    root_logger = log_config.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir,
                                                f'{save_path}.log')),
            log_config.StreamHandler()
        ]
    )

    logger = log_config.getLogger()

    # Device Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if 'cuda' in str(device):
        logger.info(f'Exp name: {exp_name} | Using GPU')
        device = 'cuda'
    else:
        logger.info(f'Exp name: {exp_name} | Using CPU. Set --cuda flag to use GPU')
        device = 'cpu'
    logger.info(f'{args.__dict__}')
    # Default audio frontend Hyperparameters setup for TAU-ASC
    frontend_params = {
        'sample_rate': 48000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000}

    num_class = 10
    if dataset == 'ESC-50':
        frontend_params['sample_rate'] = 44100
        num_class = 50
    frontend = Audio_Frontend(**frontend_params)

    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # [1] Select a CIL method
    logger.info(f"[1] Select a CIL method ({args.mode})")
    if args.mem_manage == 'uncertainty':
        logger.info(f"Select uncertainty measure approach ({args.uncert_metric})")
    loss_func = get_loss_func('clip_ce')
    if model_name == 'baseline':
        model = Baseline_CNN(num_class=num_class, frontend=frontend)
    elif model_name == 'BC-ResNet':
        model = BCResNet_Mod(num_class=num_class, frontend=frontend)
    else:
        raise Exception
    method = get_methods(
        args, loss_func, device, num_class, model
    )

    # Incrementally training
    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)
    start_time = time.time()

    # start to train each tasks
    logger.info(f'Audio frontend param:\n{frontend_params}\n')
    logger.info(f'Model:\n{model}\n')
    logger.info(f"Exp: {exp_name} | batch_size: {batch_size} | learning_rate: {learning_rate} | dataset: {dataset}")
    for cur_iter in range(args.n_tasks):
        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")

        logger.info("[2-1] Prepare a datalist for the current task")
        task_acc = 0.0
        if args.dataset == "ESC-50":
            cur_train_datalist = get_train_datalist(args, cur_iter)
            cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)
            fold_acc = 0.0
            for test_fold in range(1, 6):
                logger.info(f"Set the test fold number {test_fold} of the current task")
                method.set_current_dataset(cur_train_datalist[test_fold - 1], cur_test_datalist[test_fold - 1])
                # Increment known class for current task iteration.
                method.before_task(datalist=cur_train_datalist[test_fold - 1], init_opt=True)
                logger.info(f"[2-3] Start to train")
                fold_acc += method.train(
                    n_epoch=args.epoch,
                    batch_size=args.batch_size,
                    n_worker=8,
                )
                logger.info("[2-4] Update the information for the current task")
                method.after_task(cur_iter)
            task_acc = fold_acc / 5
        else:
            # get datalist
            cur_train_datalist = get_train_datalist(args, cur_iter)
            cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)
            logger.info("[2-2] Set environment for the current task")
            method.set_current_dataset(cur_train_datalist, cur_test_datalist)
            # Increment known class for current task iteration.
            method.before_task(datalist=cur_train_datalist, init_opt=True)

            logger.info(f"[2-3] Start to train")
            task_acc = method.train(
                n_epoch=args.epoch,
                batch_size=args.batch_size,
                n_worker=8,
            )
            logger.info("[2-4] Update the information for the current task")
            method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)

        if cur_iter > 0:
            task_records["bwt_list"].append(np.mean(
                [task_records["task_acc"][i + 1] - task_records["task_acc"][i] for i in
                 range(len(task_records["task_acc"]) - 1)]))
        logger.info("[2-5] Report task result")
    np.save(f"{log_dir}/{save_path}.npy", task_records["task_acc"])
    # Total time (T)
    duration = time.time() - start_time
    # Accuracy(A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    logger.info(f"======== Summary =======")
    logger.info(f"Total time {duration}, Avg: {duration / args.n_tasks}s")
    logger.info(f'BWT: {np.mean(task_records["bwt_list"])}, std: {np.std(task_records["bwt_list"])}')
    logger.info(f"A_last {A_last} | A_avg {A_avg}")
