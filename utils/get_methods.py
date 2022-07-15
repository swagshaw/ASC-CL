"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/7/15 下午10:56
@Author  : Yang "Jan" Xiao 
@Description : get_methods
"""
import logging
from methods.base import BaseMethod

logger = logging.getLogger()


def get_methods(args, criterion, device, n_classes, model):
    kwargs = vars(args)
    if args.mode == "finetune":
        method = BaseMethod(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            model=model,
            **kwargs,
        )
    elif args.mode == "replay":
        method = BaseMethod(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            model=model,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Choose the args.mode in "
            "[finetune, replay]"
        )
    logger.info(f"CIL Scenario: {args.mode}")
    print(f"\nn_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")

    return method
