"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/19 下午4:38
@Author  : Yang "Jan" Xiao 
@Description : generate_json
"""
import argparse
import csv
import glob
import json
import os
import random


def read_csv(path):
    csv_dict = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_dict[row['File']] = row['Category']
    return csv_dict


MSoS_class = ['Effects', 'Human', 'Music', 'Nature', 'Urban']


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="/home/xiaoyang/Dev/asc-continual-learning/data", type=str,
                        help="The path of dataset")
    parser.add_argument("--seed", type=int, default=3, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="MSoS", help="[TAU-ASC, MSoS]")
    parser.add_argument("--n_tasks", type=int, default=1, help="The number of tasks")
    parser.add_argument("--n_cls_a_task", type=int, default=0, help="The number of class of each task")
    parser.add_argument("--n_init_cls", type=int, default=5, help="The number of classes of initial task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="train", help="[train, test]")
    args = parser.parse_args()
    print(f"[0] Start to generate the {args.n_tasks} tasks of {args.dataset}.")
    if args.dataset == "MSoS":
        class_list = MSoS_class
        if args.mode == 'train':
            data_list = glob.glob('MSoS/Development/*/*.wav')
            csv_dict = read_csv('MSoS/Logsheet_Development.csv')
            print(csv_dict.popitem())
        elif args.mode == 'test':
            data_list = glob.glob('MSoS/Evaluation/*.wav')
            csv_dict = read_csv('MSoS/Logsheet_EvaluationMaster.csv')
        else:
            raise Exception
    # elif args.dataset == "TAU-ASC": TODO

    random.seed(args.seed)
    random.shuffle(class_list)
    total_list = []
    for i in range(args.n_tasks):
        if i == 0:
            t_list = []
            for j in range(args.n_init_cls):
                t_list.append(class_list[j])
            total_list.append(t_list)
        else:
            t_list = []
            for j in range(args.n_cls_a_task):
                t_list.append((class_list[j + args.n_init_cls + (i - 1) * args.n_cls_a_task]))
            total_list.append(t_list)

    print(total_list)
    label_list = []
    for i in range(len(total_list)):
        class_list = total_list[i]
        label_list = label_list + class_list
        if args.mode == 'train':
            collection_name = "collection/{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, mode='train', exp=args.exp_name, rnd=args.seed, n_cls=args.n_cls_a_task,
                iter=i
            )
        else:
            collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls_a_task, iter=i
            )
        f = open(collection_name, 'w')
        class_encoding = {category: index for index, category in enumerate(label_list)}
        dataset_list = []
        for audio_path in data_list:
            audio_name = os.path.split(audio_path)[-1]
            category = csv_dict.get(audio_name)
            if category in class_list:
                dataset_list.append([os.path.join(args.dpath, audio_path), category, class_encoding.get(category)])
        res = [{"category": item[1], "file_name": item[0], "label": item[2]} for item in dataset_list]
        print("Task ID is {}".format(i))
        print("Total samples are {}".format(len(res)))
        f.write(json.dumps(res))
        f.close()


if __name__ == "__main__":
    main()
