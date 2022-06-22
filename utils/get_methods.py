"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/19 下午2:53
@Author  : Yang "Jan" Xiao 
@Description : get_methods
"""
import logging

from methods.finetune import Finetune
from methods.rainbow import RM

logger = logging.getLogger()


def get_methods(args, criterion, device, n_classes, model):
    kwargs = vars(args)
    if args.mode == "finetune":
        method = Finetune(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            model=model,
            **kwargs,
        )
    elif args.mode == "random":
        method = Finetune(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            model=model,
            **kwargs,
        )
    elif args.mode == "rainbow":
        method = RM(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            model=model,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Choose the args.mode in "
            "[finetune, random,rainbow]"
        )
    logger.info(f"CIL Scenario: {args.mode}")
    print(f"\nn_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")

    return method
