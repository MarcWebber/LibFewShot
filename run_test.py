# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test



# # 5-1 fcn
# PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-1-Jan-04-2024-17-31-25"
PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-1-fcn"

# 5-1 grid
# PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-1-Jan-05-2024-12-35-14"

# 5-1 fcn qpth
PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-1-Jan-05-2024-12-42-27"
# PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-1-Jan-04-2024-12-10-53"
VAR_DICT = {
    "test_epoch": 10,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 100,
    "episode_size": 1,
}



# 5-5 fcn

# PATH = "./results/DeepEMD-miniImageNet--ravi-resnet12_emd-5-5-fcn"

# for 5-5 other-wise it will run quite slowly, recommend to use smaller test_episode and test_epoch, in fact 50-10 is quite large
# VAR_DICT = {
#     "test_epoch": 10,
#     "device_ids": "0",
#     "n_gpu": 1,
#     "test_episode": 50,
#     "episode_size": 1,
# }


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
