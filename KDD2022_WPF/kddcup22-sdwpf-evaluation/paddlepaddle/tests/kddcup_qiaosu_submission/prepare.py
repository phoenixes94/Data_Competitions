# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import paddle

def prep_env():
    settings = {
    "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
    "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
    "data_path": "./data",
    "filename": "sdwpf_baidukddcup2022_full.csv",
    "task": "MS",
    "target": "Patv",
    "checkpoints": "checkpoints",
    "input_len": 288,
    "output_len": 288,
    "num_in_cluster":5,
    "start_col": 3,
    "day_len": 144,
    "nlayers": 3,
    "dropout": 0.4,
    "model_dim": 6,
    "model": "trm",
    "num_workers": 5,
    "train_epochs": 10,
    "gpu": 0,
    "capacity": 134,
    "pred_file":"predict.py",
    "framework": "paddlepaddle",
    "model_root_dir": "checkpoints",
    "json_root_dir": "mean_std_json",
    "other_del_feature": ["Tmstamp", "Wdir", "Etmp", "Itmp", "Ndir"],
    "help_turbine_dict_json_dir": "./turbine_cluster/help_turbine_dict.json",
    "cluster_json_dir": "./turbine_cluster/turbine_cluster_list.json",
    "is_debug": False
    }

#     settings = {
#         "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
#         "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
#         "data_path": "./data",
#         "filename": "sdwpf_baidukddcup2022_full.csv",
#         "task": "MS",
#         "target": "Patv",
#         "checkpoints": "checkpoints",
#         "input_len": 288,
#         "output_len": 288,
#         "start_col": 0,
#         "day_len": 144,
#         "dropout": 0.0,
#         "model_dim":12,
#         "model_name":"trm",
#         "num_workers": 5,
#         "capacity": 134,
#         "pred_file": "predict.py",
#         "framework": "pytorch",
#         "is_debug": True
#     }
#     ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')
    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
