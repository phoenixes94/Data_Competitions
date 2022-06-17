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
from wpf_model import WPFModel
# import optimization as optim
# from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model
# import matplotlib.pyplot as plt
import pickle


def load_data():
    now_abs_dir = os.path.dirname(os.path.realpath(__file__))
    # load offline data
    print(os.path.join(now_abs_dir, "data_mean.pkl"))
    with open(os.path.join(now_abs_dir,  "data_mean.pkl"), "rb") as g:
        data_mean = pickle.load(g)

    with open(os.path.join(now_abs_dir, "data_scale.pkl"), "rb") as p:
        data_scale = pickle.load(p)

    with open(os.path.join(now_abs_dir,  "edges.pkl"), "rb") as q:
        edges = pickle.load(q)

    return data_mean, data_scale, edges


@paddle.no_grad()
def predict(settings, index):  # , valid_data, test_data):
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
    _data_mean, _data_scale, edges = load_data()
    
    data_mean = paddle.to_tensor(_data_mean, dtype="float32")
    data_scale = paddle.to_tensor(_data_scale, dtype="float32")

    graph = pgl.Graph(num_nodes=134, edges=edges)
    graph = graph.tensor()

    print(settings["model_{}".format(index)])
    model = WPFModel(config=settings["model_{}".format(index)])

    print(os.path.join(settings["checkpoints"], "{}".format(index), "checkpoint"))
    global_step = load_model(os.path.join(settings["checkpoints"], "{}".format(index), "checkpoint"), model)
    model.eval()

    test_x_f = settings["path_to_test_x"]
    test_x_ds = TestPGL4WPFDataset(filename=test_x_f)
    test_x = paddle.to_tensor(test_x_ds.get_data()[:, :, -settings["input_len"]:, :], dtype="float32")
    test_y = paddle.ones(shape=[1, 134, settings["output_len"], 12], dtype="float32")

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
    # i = 1
    prediction = [0 for i in range(settings["num_model"])]
    for i in range(settings["num_model"]):
        prediction[i] = predict(settings, i)  # , valid_data, test_data)

    predictions = ( 1.0 * prediction[0] + 1.0 * prediction[1] + 1.0 * prediction[2]) / 3

    # predictions = prediction[1]
    print(predictions.shape)
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
