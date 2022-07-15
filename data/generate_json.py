"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/19 下午4:38
@Author  : Yang "Jan" Xiao 
@Description : generate_json
"""
import argparse
import json
import random
import os

import pandas as pd
from soundata.datasets import tau2019uas
from tqdm import tqdm

os.chdir("..")
print(os.getcwd())

TAU_class = ['airport', 'bus', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'metro_station', 'metro',
             'public_square', 'tram', 'park']


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="/home/xiaoyang/Dev/asc-continual-learning/data", type=str,
                        help="The path of dataset")
    parser.add_argument("--seed", type=int, default=3, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="TAU-ASC", help="[TAU-ASC, ESC-50]")
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--n_cls_a_task", type=int, default=2, help="The number of class of each task")
    parser.add_argument("--n_init_cls", type=int, default=2, help="The number of classes of initial task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="test", help="[train, test]")
    args = parser.parse_args()

    print(f"[0] Start to generate the {args.n_tasks} tasks of {args.dataset}.")
    if args.dataset == "TAU-ASC":
        class_list = TAU_class
        tau_dataset = tau2019uas.Dataset(data_home='data/TAU_ASC')
        clip_ids = tau_dataset.clip_ids
        dev_clip_ids = [id for id in clip_ids if 'development' in id]
        all_clips = tau_dataset.load_clips()
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
                clip_ids = [id for id in dev_clip_ids if all_clips[id].split == 'development.train']
            else:
                collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                    dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls_a_task, iter=i
                )
                clip_ids = [id for id in dev_clip_ids if all_clips[id].split == 'development.test']
            f = open(os.path.join(args.dpath, collection_name), 'w')
            class_encoding = {category: index for index, category in enumerate(label_list)}
            dataset_list = []
            for id in clip_ids:
                clip = all_clips.get(id)
                tag = clip.tags.labels[0]
                if tag in class_list:
                    dataset_list.append([id, tag, class_encoding.get(tag)])
            res = [{"tag": item[1], "audio_name": item[0], "label": item[2]} for item in dataset_list]
            print("Task ID is {}".format(i))
            print("Total samples are {}".format(len(res)))
            f.write(json.dumps(res))
            f.close()
    elif args.dataset == "ESC-50":
        data_list = []
        meta = pd.read_csv(os.path.join(args.dpath, 'ESC-50-master/meta/esc50.csv'))
        for test_fold_num in range(1, 6):
            if args.mode == 'train':
                data_list = meta[meta['fold'] != test_fold_num]
            elif args.mode == 'test':
                data_list = meta[meta['fold'] == test_fold_num]
            print(f'ESC-50 {args.mode} set using fold {test_fold_num} is creating, using sample rate {44100} Hz ...')
            class_list = sorted(data_list["category"].unique())
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
                    collection_name = "collection/{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}" \
                                      "_task{iter}_{test_fold_num}.json".format(dataset=args.dataset, mode='train',
                                                                                exp=args.exp_name, rnd=args.seed,
                                                                                n_cls=args.n_cls_a_task,
                                                                                iter=i, test_fold_num=test_fold_num
                                                                                )

                else:
                    collection_name = "collection/{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}" \
                                      "_{test_fold_num}.json".format(dataset=args.dataset, rnd=args.seed,
                                                                     n_cls=args.n_cls_a_task, iter=i,
                                                                     test_fold_num=test_fold_num
                                                                     )
                f = open(os.path.join(args.dpath, collection_name), 'w')
                class_encoding = {category: index for index, category in enumerate(label_list)}
                dataset_list = []

                for index in tqdm(range(len(data_list))):
                    row = data_list.iloc[index]
                    file_path = os.path.join(args.dpath, 'ESC-50-master', 'audio', row["filename"])
                    if row['category'] in class_list:
                        dataset_list.append([file_path, row['category'], class_encoding.get(row['category'])])
                res = [{"tag": item[1], "audio_name": item[0], "label": item[2]} for item in dataset_list]
                print("Task ID is {}".format(i))
                print("Total samples are {}".format(len(res)))
                f.write(json.dumps(res))
                f.close()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
