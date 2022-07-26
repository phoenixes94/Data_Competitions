# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import joblib
from matplotlib.pyplot import axis
import paddle
import paddle.nn.functional as F
import tqdm
import yaml
import numpy as np
from easydict import EasyDict as edict

import pgl
from pgl.utils.logger import log
from paddle.io import DataLoader
# import random

from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
# import optimization as optim
# from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model
# import matplotlib.pyplot as plt
import pickle


def load_data(index):
    now_abs_dir = os.path.dirname(os.path.realpath(__file__))
    # load offline data
    print(os.path.join(now_abs_dir, "checkpoints", index,  "data_mean.pkl"))
    with open(os.path.join(now_abs_dir, "checkpoints", index, "data_mean.pkl"), "rb") as g:
        data_mean = pickle.load(g)

    with open(os.path.join(now_abs_dir, "checkpoints", index, "data_scale.pkl"), "rb") as p:
        data_scale = pickle.load(p)

    with open(os.path.join(now_abs_dir, "checkpoints", index, "edges.pkl"), "rb") as q:
        edges = pickle.load(q)

    return data_mean, data_scale, edges


@paddle.no_grad()
def predict(settings, index, test_x_ds):  # , valid_data, test_data):
    """
    Desc:
        Forecasting the wind power in a naive manner
    Args:
        settings:
    Returns:
        pred_y: [1, 134, 288]
    """
    # data_mean：[1, 134, 1, 1]
    # data_scale：[1, 134, 1, 1]
    _data_mean, _data_scale, edges = load_data(str(index))
    
    data_mean = paddle.to_tensor(_data_mean, dtype="float32")
    data_scale = paddle.to_tensor(_data_scale, dtype="float32")

    graph = pgl.Graph(num_nodes=134, edges=edges)
    graph = graph.tensor()

    turb_setting = settings["model_{}".format(index)]
    print(turb_setting)
    if index in [6, 7, 8]:
        from wpf_model_st import WPFModel
    elif index in [9, 10]:
        from wpf_model_ac import WPFModel
    else:
        from wpf_model import WPFModel
    model = WPFModel(config=turb_setting)

    print(os.path.join(settings["checkpoints"], "{}".format(index), "checkpoint"))
    global_step = load_model(os.path.join(settings["checkpoints"], "{}".format(index), "checkpoint"), model)
    model.eval()

    test_x = paddle.to_tensor(test_x_ds.get_data()[:, :,  -turb_setting["input_len"]:, :], dtype="float32")
    test_y = paddle.ones(shape=[1, 134, turb_setting["output_len"], 12], dtype="float32")

    # [1, 134, 288]
    # test_x:[1, 134, 144, 12]; test_y:[1, 134, 288, 12]; data_mean:[1, 134, 1, 1]
    pred_y = model(test_x, test_y, data_mean, data_scale, graph)
    pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :,
                                                                 -1])
    print(pred_y)
    pred_y = np.transpose(pred_y, [
        1,
        2,
        0,
    ])
    pred_y = pred_y.numpy()

    return pred_y


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    # data process for save time
    test_x_f = settings["path_to_test_x"]
    test_x_ds_full = TestPGL4WPFDataset(filename=test_x_f, del_feat_ind=0)
    test_x_ds_del_ptrv = TestPGL4WPFDataset(filename=test_x_f, del_feat_ind=1)
    test_x_ds_clear_del4p = TestPGL4WPFDataset(filename=test_x_f, del_feat_ind=3)
    test_x_ds_clear_delp = TestPGL4WPFDataset(filename=test_x_f, del_feat_ind=5)

    prediction = [0 for _ in range(settings["num_model"])]
    for i in range(settings["num_model"]):
        if i in [0, 2]:
            test_x_ds = test_x_ds_full
        elif i in [1,]:
            test_x_ds = test_x_ds_del_ptrv
        elif i in [3, 4, 8]:
            test_x_ds = test_x_ds_clear_del4p
        else:
            test_x_ds = test_x_ds_clear_delp
        prediction[i] = predict(settings, i, test_x_ds)
    
    predictions = (1.35 * prediction[0] + 0.55 * prediction[1] + 1.1 * prediction[2] +\
                    1.0 * prediction[3] + 1.0 * prediction[4] + 1.0 * prediction[5] + \
                    1.0 * prediction[6] + 1.0 * prediction[7] + 1.0 * prediction[8] + \
                    1.0 * prediction[9] + 1.0 * prediction[10]) / 11

    # predictions = predict(settings, 10, test_x_ds_clear_delp)
    # predictions = prediction[6]

    # RMSE: 47.17026989334526, MAE: 39.486272397359414, Score: 43.32827114535233, and Accuracy: 60.6367%
    # predictions = ( 1 * prediction[0] + 1 * prediction[1] + 1 * prediction[2] + \
    #                 1 * prediction[3] + 1 * prediction[3] + 1 * prediction[3] + 1 * prediction[3]) / 7

    # RMSE: 47.22517822241438, MAE: 39.24244758937393, Score: 43.233812905894155, and Accuracy: 60.3170%
    # predictions = ( 1 * prediction[0] + 1 * prediction[1] + 1 * prediction[2] + \
    #                 1 * prediction[3] + 1 * prediction[3] + 1 * prediction[3]) / 6

    # 全部6个
    #  RMSE: 47.930290857729105, MAE: 38.44831633803584, Score: 43.18930359788247, and Accuracy: 58.5734%
    # predictions = ( 1 * prediction[0] + 1 * prediction[1] + 1 * prediction[2] + \
    #                 1 * prediction[3] + 1 * prediction[4] + 1 * prediction[5]) / 6

    # 全部5个
    # RMSE: 47.81502660818225, MAE: 38.5331096761427, Score: 43.174068142162476, and Accuracy: 58.8305%
    # predictions = ( 1 * prediction[0] + 1 * prediction[1] + 1 * prediction[2] + \
    #                 1 * prediction[3] + 1 * prediction[4] + 1 * prediction[5] + 1 * prediction[6] + 1 * prediction[7] + \
    #                 1 * prediction[8] + 1 * prediction[9]) / 10

    # predictions = np.concatenate((predict(settings, 1)[:, :144, :], predict(settings, 4)[:, 144:, :]), axis=1)
    # predictions = (1.25 * predict(settings, 0) + 0.75 * predict(settings, 1) + 1 * predict(settings, 4) ) / 3

    # predictions = (1.35 * predict(settings, 0) + 0.55 * predict(settings, 1) + 1.1 * predict(settings, 2) +\
    #                 1.0 * predict(settings, 3) + 1.0 * predict(settings, 4) ) / 5
    # predictions = np.concatenate( ((predictions[:, :144, :] + predict(settings, 1)[:, :144, :])/2, predictions[:, 144:, :]), axis=1) 

    # predictions = predict(settings, 0)[:, :144, :144]
    print('predictions.shape: ', predictions.shape)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    # print(config)
    # size = [config.input_len, config.output_len]
    # train_data = PGL4WPFDataset(
    #     config.data_path,
    #     filename=config.filename,
    #     size=size,
    #     flag='train',
    #     total_days=config.total_days,
    #     train_days=config.train_days,
    #     val_days=config.val_days,
    #     test_days=config.test_days)

    predict(config)  # , valid_data, test_data)
