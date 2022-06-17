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
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "./data",
        "filename": "sdwpf_baidukddcup2022_full.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size": 214,
        "val_size": 31,
        "total_size": 245,
        # "lstm_layer": 2,
        # 'min_distinct_ratio': 0.1, 
        # 'min_non_zero_ratio': 0.5
        # "dropout": 0.05,
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True,
        "input_len": 144,
        "output_len": 288,
        "model_0":{
            "input_len": 144,
            "output_len": 288,
            "capacity": 134,
            # add para
            "var_len": 10,
            "hidden_dims": 128,
            "dropout": 0.5,
            "nhead": 8,
            "encoder_layers": 2,
            "decoder_layers": 2,
        },
        "model_1":{
            "input_len": 144,
            "output_len": 288,
            "capacity": 134,
            # add para
            "var_len": 10,
            "hidden_dims": 128,
            "dropout": 0.5,
            "nhead": 4,
            "encoder_layers": 3,
            "decoder_layers": 1,
        },
        "model_2":{
            "input_len": 144,
            "output_len": 288,
            "capacity": 134,
            # add para
            "var_len": 10,
            "hidden_dims": 128,
            "dropout": 0.5,
            "nhead": 8,
            "encoder_layers": 2,
            "decoder_layers": 2,
        },
        "num_model": 3,
    }
    ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings