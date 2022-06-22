"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/19 下午8:51
@Author  : Yang "Jan" Xiao 
@Description : finetune
"""
from methods.base import BaseMethod


class Finetune(BaseMethod):
    def __init__(self, criterion, device, n_classes, **kwargs):
        super().__init__(criterion, device, n_classes, **kwargs)
        if self.mode == "finetune":
            self.memory_size = 0

        if self.mode == "random" and self.mem_manage == "default":
            self.mem_manage = "random"
