# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A demo of the forecasting method
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/18
"""
#import matplotlib.pyplot as plt
import os
import time
import numpy as np

import json
import pandas as pd
import joblib
import paddle

import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path

class TestData(object):
    """
        Desc: Test Data
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,       # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data = deepcopy(self.df_raw)
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]
            data = self.df_data[cols]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]
        df = self.df_raw[border1:border2]
        return seq, df

    def get_all_turbines(self):
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs

# jupyter 那些数据处理流程
def process_data(tid,mean_std_dir,data):
    # 0 test_data.py非常粗暴的将nan填充了0，还原成nan，用步骤3上下文填充
    col = list(data.columns)
    col.remove("Tmstamp")
    nan_idx = (data==0).sum(1)>3
    if np.sum(nan_idx)!=0:
        pass
        # print(data.loc[nan_idx,col])
    data.loc[nan_idx,col]=np.nan
    # 1时间编码
    time_set = np.unique(data["Tmstamp"].values)
    time_set = sorted(time_set)
    #print(time_set)
    assert len(time_set) == 144
    time_dict = {}
    for i in range(len(time_set)):
        time_dict[time_set[i]] = i
    data["Tmstamp"] = data["Tmstamp"].replace(time_dict)
    #print(time_dict)
    # 3填充空值
    data = data.fillna(method="ffill")
    data = data.fillna(method="bfill")
    #4官方标注的异常值
    # (1)"patv"<0=========>=0
    idx1 = data["Patv"] < 0
    if np.sum(data.loc[idx1, ["Patv"]].values < -20) > 0:
        raise NameError("有负值小于-20,{}".format(data.loc[data["Patv"] < -20, ["Patv"]]))
    data.loc[idx1, ["Patv"]] = 0
    # (2)Patv<=1 AND Wspd>2.5
    # 因为这块没法做异常检测，所以阈值卡多一点
    idx2 = ((data['Patv'] < 100) & (data['Wspd'] > 2.5))
    #(3)修正其他数值
    idx3_1 = data['Pab1'] > 89
    data.loc[idx3_1, ["Pab1"]] = 89
    idx3_2 = data['Pab2'] > 89
    data.loc[idx3_2, ["Pab2"]] = 89
    idx3_3 = data['Pab3'] > 89
    data.loc[idx3_3, ["Pab3"]] = 89
    idx4 = data['Wdir'] < -180
    data.loc[idx4, ["Wdir"]] = -180
    idx5 = data['Wdir'] > 180
    data.loc[idx5, ["Wdir"]] = 180
    idx6 = data['Ndir'] < -720
    data.loc[idx6, ["Ndir"]] = -720
    idx7 = data['Ndir'] > 720
    data.loc[idx7, ["Ndir"]] = 720
    # 异常检测pass
    # 异常修正(idx2,)=====================
    # linear_repair_dir = os.path.join(mean_std_dir,"linear_repair.json")
    # with open(linear_repair_dir, "r", encoding="utf-8") as f:
    #     linear_repair = json.load(f)
    # # print(settings["turbine_id"], linear_repair)
    # x3 = data.loc[idx2, ["Wspd"]].values**3
    # x2 = data.loc[idx2, ["Wspd"]].values**2
    # x1 = data.loc[idx2, ["Wspd"]].values
    # K = linear_repair[str(tid)]["K"]
    # B = linear_repair[str(tid)]["B"]
    # MAX = linear_repair[str(tid)]["MAX_P"]
    # MIN = 0
    # prey = K[0]*x3+K[1]*x2+K[2]*x1+B
    # # plt.scatter(x1,prey)
    # # plt.show()
    # max_idx = prey > MAX
    # min_idx = prey < MIN
    # prey[max_idx] = MAX
    # prey[min_idx] = MIN
    # data.loc[idx2, ["Patv"]]=prey
    #plt.scatter(data["Wspd"],data["Patv"])
    #plt.show()
    return data

def scale_func(data,mean_std):
    for c in data.columns:
        m = mean_std[c]["mean"]
        s = mean_std[c]["std"]
        data[c] = (data[c].values-m)/s
    return data


@paddle.no_grad()
def forecast_one(test_turbines,turbine_cluster,help_turbine_dict,args):
    # 便利当前风机预测需要的风机数据
    help_and_target_turbine_list = turbine_cluster.copy()
    for turbid in turbine_cluster:
        help_turbine_list = help_turbine_dict[str(turbid)]
        print(turbid, help_turbine_list)
        for help_turbine in help_turbine_list:
            if help_turbine not in help_and_target_turbine_list:
                help_and_target_turbine_list.append(help_turbine)
    args["help_and_target_turbine_list"] =  help_and_target_turbine_list
    args["num_in_cluster"] = len(turbine_cluster)

    file_name = [str(i) for i in turbine_cluster]
    file_name = "_".join(file_name) + "_model"
    print(file_name)
    path_to_model = os.path.join(args["model_root_dir"],args["model"],file_name)
    # model.load_state_dict(torch.load(path_to_model))
    model = paddle.jit.load(path_to_model)
    model.eval()

    X_list = []
    for tid in help_and_target_turbine_list:
        json_dir = os.path.join(args["json_root_dir"],
                                "turb{}.json".format(tid))
        with open(json_dir, "r", encoding="utf-8") as f:
            mean_std = json.load(f)
        _, raw_test_x_data = test_turbines.get_turbine(tid-1)
        # 去掉turb编号
        test_x_data = raw_test_x_data.drop(["TurbID","Day"], axis=1)
        # 处理数据格式,传入场站号和保存json的根目录
        test_x_data = process_data(tid,args["json_root_dir"], test_x_data)
        test_x_data = test_x_data.drop(args["other_del_feature"],axis=1)
        # 标准化
        test_x_data = scale_func(test_x_data,mean_std)
        test_x_data = test_x_data.values
        test_x_data = np.expand_dims(test_x_data,axis=1)
        X_list.append(test_x_data)
    test_x_data = np.concatenate(X_list,axis=1)

    test_x_data = paddle.to_tensor(test_x_data)
    last_observ = test_x_data[-args["input_len"]:]
    last_observ = paddle.unsqueeze(last_observ,0)
    last_observ = paddle.cast(paddle.transpose(last_observ,perm=(1, 0, 2, 3)),'float32')

    # print(last_observ.shape)
    prediction = model(last_observ)[0]
    # print(f'output shape: {prediction.shape}')
    # 格式要求最后一维是有1维的
    prediction = prediction.numpy()[:,None]
    
    zero_idx = prediction<0
    prediction[zero_idx]=0
    #===========
    assert prediction.shape[1]==1
    # tmp = np.zeros([288,1])
    # # 插值
    # S = args.sample_step
    # for i in range(S):
    #     tmp[i::S, 0] = prediction[:, 0]
    # prediction = tmp

    # if np.std(prediction[:,0])<0.1:
    #     prediction += np.random.random(prediction.shape)*0.001

    # 满足标准差要求
    #print(prediction)
    #===========
    prediction_dict = {}
    # 输出的是平均值

    for idx, turbid in enumerate(turbine_cluster):
        prediction_dict[turbid] = prediction[:, 0]
    return prediction_dict


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    checkpoints_path = Path(settings['checkpoints'])
    abs_dir = checkpoints_path.parent.absolute()
    for k,v in settings.items():
        if 'dir' in k:
            settings[k] = str(abs_dir/v)

    start_time = time.time()
    predictions = []

    settings["turbine_id"] = 0
    # # train_data = Experiment.train_data
    # train_data = exp.load_train_data()
    # train_data 纯粹是为了算scaler，我用加载json的
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time
    # 返回的是一个类的对象
    test_x = TestData(path_to_data=settings["path_to_test_x"], farm_capacity=settings["capacity"])
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    # 读取风机分组和辅助场站json===========================================================
    turbine_cluster_dir = settings['cluster_json_dir'] #os.path.join(settings["json_root_dir"], "turbine_cluster_list.json")
    help_turbine_dir = settings['help_turbine_dict_json_dir'] #os.path.join(settings["json_root_dir"], "help_turbine_dict.json")
    with open(turbine_cluster_dir, "r", encoding="utf-8") as f:
        turbine_cluster_list = json.load(f)
    with open(help_turbine_dir, "r", encoding="utf-8") as f:
        help_turbine_dict = json.load(f)
    #print(len(turbine_cluster_list),turbine_cluster_list)
    predictions = np.zeros([134,288])
    for turbine_cluster in turbine_cluster_list:
        # question 为了能跑 这里搞成了0
        # print('\n>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
        prediction_dict = forecast_one(test_x,turbine_cluster,help_turbine_dict, settings)
        for turbine_id,pre in prediction_dict.items():
            predictions[turbine_id-1] = pre
    # tmp (134,288,1)
    tmp = np.expand_dims(predictions,2)
    # for i in range(len(tmp)):
    #     L = tmp.shape[1]
    #     rand = np.random.rand(L)
    #     # print(tmp[i,:10,0])
    #     tmp[i,:,0] = tmp[i,:,0]+rand*0.001
    #     # print(tmp[i, :10, 0])
    #     # print("-")
    #raise NameError(np.isnan(tmp[:,:,-1]).sum(),(tmp[:,:,-1]<0).sum())
    return tmp*1000

def performance(settings, idx, prediction, ground_truth, ground_truth_df):
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
        idx:
        prediction:
        ground_truth:
        ground_truth_df:
    Returns:
        MAE, RMSE and Accuracy
    """
    overall_mae, overall_rmse, _, overall_latest_rmse = \
        metrics.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    if overall_latest_rmse < 0:
        raise Exception("The RMSE of the last 24 hours is negative ({}) in the {}-th prediction"
                              "".format(overall_latest_rmse, idx))
    acc = 1 - overall_latest_rmse / settings["capacity"]
    return overall_mae, overall_rmse, acc

if __name__=='__main__':
    from main import get_args_parser, check_and_update_args
    from util import metrics
    parser = get_args_parser()
    args = parser.parse_args()
    args = check_and_update_args(args)
    args["model_root_dir"] = f'{args["model_root_dir"]}/{args["model"]}'
    print(args['capacity'])

    args.path_to_test_x = 'predict_data/test_x/0001in.csv'
    args.path_to_test_y = 'predict_data/test_y/0001out.csv'
    test_data = TestData(path_to_data=args["path_to_test_y"], farm_capacity=args["capacity"])
    turbines, raw_turbines = test_data.get_all_turbines()
    test_ys = []
    for turbine in turbines:
        test_ys.append(turbine[:args["output_len"], -1:])

    gt_ys = np.array(test_ys)
    gt_turbines = raw_turbines

    prediction = forecast(args)
    tmp_mae, tmp_rmse, tmp_acc = performance(args, 0, prediction, gt_ys, gt_turbines)
    print('\n\tThe {}-th prediction -- '
            'RMSE: {}, MAE: {}, Score: {}, '
            'and Accuracy: {:.4f}%'.format(0, tmp_rmse, tmp_mae, (tmp_rmse + tmp_mae) / 2, tmp_acc * 100))