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
import time
import datetime
import numpy as np
import pandas as pd
# import paddle
import joblib
from paddle.io import Dataset

import pgl
from pgl.utils.logger import log
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj


def time2int(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    time_int = int(time.mktime(data_sj))
    return time_int


def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')


def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e


def func_add_h(x):
    time_obj = time2obj(x)
    hour_e = time_obj.tm_hour
    return hour_e


def LOF(data, outliers_fraction=0.05):
    WIND = "Wspd"
    norm_data = data[[WIND, "Patv"]].copy()
    norm_data[WIND] = norm_data[WIND]/max(norm_data[WIND])
    norm_data["Patv"] = norm_data["Patv"]/max(norm_data["Patv"])

    # fit the model
    clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
    y_pred = clf.fit_predict(norm_data)

    good_idx = y_pred == 1
    bad_idx = y_pred == -1

    # =============plot=================
    good_data = data.loc[good_idx, :]
    bad_data = data.loc[bad_idx, :]
    print("good_num:{} bad_num:{}".format(len(good_data), len(bad_data)))

    return good_data, bad_data


def pair_data(bad_data, output_path):
    print(bad_data.columns)
    x_col = ['Wspd']

    gbm = joblib.load(os.path.join(output_path, "lgb_model_p0630_20.pkl"))
    log.info('lgb model loaded!')
    pre_Y = gbm.predict(bad_data[x_col])
    bad_data["Patv"] = pre_Y

    return bad_data


def preprocess_data(data, output_path):
    col = list(data.columns)
    nan_idx = (data == 0).sum(1) > 3
    if np.sum(nan_idx) != 0:
        pass
        # print(data.loc[nan_idx,col])
    data.loc[nan_idx, col] = np.nan

    # 1.填充空值
    data = data.fillna(method="ffill")
    data = data.fillna(method="bfill")
    # 4官方标注的异常值
    # (1)"patv"<0=========>=0
    idx1 = data["Patv"] < 0
    if np.sum(data.loc[idx1, ["Patv"]].values < -20) > 0:
        raise NameError(
            "有负值小于-20,{}".format(data.loc[data["Patv"] < -20, ["Patv"]]))
    # data.loc[idx1, ["Patv"]] = 0
    # (2)Patv<=1 AND Wspd>2.5
    # 因为这块没法做异常检测，所以阈值卡多一点
    idx2 = ((data['Patv'] < 100) & (data['Wspd'] > 2.5))
    # (3)修正其他数值
    # idx3_1 = data['Pab1'] > 89
    # data.loc[idx3_1, ["Pab1"]] = 89
    # idx3_2 = data['Pab2'] > 89
    # data.loc[idx3_2, ["Pab2"]] = 89
    # idx3_3 = data['Pab3'] > 89
    # data.loc[idx3_3, ["Pab3"]] = 89
    # idx4 = data['Wdir'] < -180
    # data.loc[idx4, ["Wdir"]] = -180
    # idx5 = data['Wdir'] > 180
    # data.loc[idx5, ["Wdir"]] = 180
    # idx6 = data['Ndir'] < -720
    # data.loc[idx6, ["Ndir"]] = -720
    # idx7 = data['Ndir'] > 720
    # data.loc[idx7, ["Ndir"]] = 720

    # 异常检测pass
    del_rate = 0.20
    not_wash_data = data.loc[idx2, :]
    wash_data = data.loc[~idx2, :]
    good_data, bad_data = LOF(wash_data, outliers_fraction=del_rate)

    # 异常修正(idx2,)=====================
    bad_data = bad_data.append(not_wash_data)
    bad_data = pair_data(bad_data, output_path)

    finally_data = good_data.append(bad_data)
    assert len(finally_data) == len(data)
    finally_data = finally_data.sort_values(by=["TurbID", "Day", "Tmstamp"], ascending=True)

    return finally_data


def isoforest(data, WIND, C=0.05, show=False):
    clf = IsolationForest(contamination=C)
    norm_data = data[[WIND, "Patv"]].copy()
    norm_data[WIND] = norm_data[WIND] / max(norm_data[WIND])
    norm_data["Patv"] = norm_data["Patv"] / max(norm_data["Patv"])
    clf.fit(norm_data)

    y_label = clf.predict(norm_data)
    good_idx = y_label == 1
    bad_idx = y_label == -1
    # =================================
    good_data = data.loc[good_idx, :]
    bad_data = data.loc[bad_idx, :]
    # print(" good_num:{} bad_num:{}".format(len(good_data), len(bad_data)))

    return bad_idx


def process_data_water(data):
    final_data = pd.DataFrame()
    data_gp = data.groupby('TurbID')
    for tid in range(1, 135):
        data = data_gp.get_group(tid)
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
        idx2 = ((data['Patv'] < 1) & (data['Wspd'] > 2.5))
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
        SHOW = False
        # 异常检测
        idx_wash = isoforest(data, WIND="Wspd", C=0.05,show=SHOW)
        #异常修正：lightgbm====================================
        model_dir = os.path.join('./repair_model', "repair_model_{}.pkl".format(tid))
        repair_model = joblib.load(model_dir)
        # col_name = list(data.columns)
        # col_name.remove("Tmstamp")
        # col_name.remove("Patv")
        # all_anomaly_idx = idx1 | idx2 |idx3_1 | idx3_2 | idx3_3 | idx4 | idx5 | idx6 |idx7
        col_name = ["Wspd"]
        idx_pre = idx2 | idx_wash
        tmp_data = data.loc[idx_pre, col_name]
        pre = repair_model.predict(tmp_data)
        data.loc[idx_pre, ["Patv"]] = pre

        final_data = pd.concat([final_data, data])

    return final_data

class PGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(
            self,
            data_path,
            filename='wtb5_10.csv',
            flag='train',
            size=None,
            capacity=134,
            day_len=24 * 6,
            train_days=153,  # 15 days
            val_days=16,  # 3 days
            test_days=15,  # 6 days
            total_days=184,  # 30 days
            theta=0.9, 
            output_path='./output/baseline/'):

        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity
        self.theta = theta
        self.output_path = output_path

        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename

        self.total_size = total_days * self.unit_size # 35280 = 245 * 144 
        self.train_size = train_days * self.unit_size # 33120 = 230 * 144
        self.val_size   = val_days * self.unit_size   # 2160 = 15 * 144

        self.test_size = test_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        raw_df_data = self.data_preprocess_feature(df_raw)
        df_data = self.data_preprocess(raw_df_data)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        data_x, graph = self.build_graph_data(df_data)
        log.info(f"data_shape: {data_x.shape}")
        log.info(f"graph: {graph}")
        self.data_x = data_x
        self.graph = graph

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]

        if self.flag == "train":
            perm = np.arange(0, seq_x.shape[0])
            np.random.shuffle(perm)
            return seq_x[perm], seq_y[perm]
        else:
            return seq_x, seq_y

    def __len__(self):
        return self.data_x.shape[1] - self.input_len - self.output_len + 1

    def data_preprocess_feature(self, df_data):
        # feature_name = [
        #     n for n in df_data.columns
        #     if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
        #     'TurbID' not in n
        # ]
        # if self.flag == 'val':
        # del ["Tmstamp","Wdir","Ndir","Etmp","Itmp",]
        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n and 'Wdir' not in n and 'Etmp' not in n and 
            'Itmp' not in n and 'Ndir' not in n and 'Prtv' not in n
        ]
        # feature_name = [
        #     n for n in df_data.columns
        #     if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
        #     'TurbID' not in n and 'Wdir' not in n and 'Etmp' not in n and 
        #     'Itmp' not in n and 'Ndir' not in n and 'Prtv' not in n 
        #     and 'Pab1' not in n and 'Pab2' not in n and 'Pab3' not in n 
        # ]
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        log.info('adding time')
        # t = df_data['Tmstamp'].apply(func_add_t)
        t = df_data['Tmstamp'].apply(lambda x: x)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)
        
        raw_df_data = new_df_data
        return raw_df_data

    def data_preprocess(self, raw_df_data):
        pd.set_option('mode.chained_assignment', None)
        new_df_data = raw_df_data.replace(
            to_replace=np.nan, value=0, inplace=False)

        return new_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        # ['weekday', 'time', 'Wspd', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]

        data = df_data.values
        data = np.reshape(data,
                          [self.capacity, self.total_size, len(cols_data)])
        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.capacity, self.total_size, len(cols_data)])

        #begin [0, 33120-144, 33120+2160-144,]
        border1s = [
            0, self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        #end [33120, 33120+2160, 33120+2160+0]
        border2s = [
            self.train_size, self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]

        self.data_mean = np.expand_dims(
            np.mean(
                data[:, border1s[0]:border2s[0], 2:],
                axis=(1, 2),
                keepdims=True),
            0)
        self.data_scale = np.expand_dims(
            np.std(data[:, border1s[0]:border2s[0], 2:],
                   axis=(1, 2),
                   keepdims=True),
            0)

        #self.data_mean = np.mean(data[:, border1s[0]:border2s[0], 2:]).reshape([1, 1, 1, 1])
        #self.data_scale = np.std(data[:, border1s[0]:border2s[0], 2:]).reshape([1, 1, 1, 1])

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data))

        data_x = data[:, border1:border2, :]
        data_edge = data[:, border1s[0]:border2s[0], -1]
        edge_w = np.corrcoef(data_edge)  # (134, 134)

        # 根据persion相关系数，计算前topK个相关的点，一共是134 * 134，最后取其中536个作为连接边
        k = 5
        topk_indices = np.argpartition(edge_w, -k, axis=1)[:, -k:]
        rows, _ = np.indices((edge_w.shape[0], k))
        kth_vals = edge_w[rows, topk_indices].min(axis=1).reshape([-1, 1])

        row, col = np.where(edge_w > kth_vals)
        edges = np.concatenate([row.reshape([-1, 1]), col.reshape([-1, 1])],
                               -1)  # (536, 2)
        print(edges.shape)
        # for online test
        if self.flag == "train":
            import pickle
            # now_abs_dir = os.path.dirname(os.path.realpath(__file__))
            with open(self.output_path + '/' + "edges.pkl", "wb") as g:
                pickle.dump(edges, g)
            log.info("edge_w.shape[0]: %s " % edge_w.shape[0])
            log.info("Edges saved!")
            
        graph = pgl.Graph(num_nodes=edge_w.shape[0], edges=edges)
        return data_x, graph


class TestPGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
    """

    def __init__(self, filename, capacity=134, day_len=24 * 6, test_x=False):

        super().__init__()
        self.unit_size = day_len

        self.start_col = 0
        self.capacity = capacity
        self.filename = filename
        self.test_x = test_x

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.filename)
        if self.test_x:
            df_raw = process_data_water(df_raw)
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        data_x = self.build_graph_data(df_data)
        self.data_x = data_x

    def data_preprocess(self, df_data):
        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n
        ]
        if self.test_x:
            feature_name = [
                n for n in df_data.columns
                if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
                'TurbID' not in n and 'Wdir' not in n and 'Etmp' not in n and 
                'Itmp' not in n and 'Ndir' not in n and 'Prtv' not in n
            ]
            # feature_name = [
            #     n for n in df_data.columns
            #     if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            #     'TurbID' not in n and 'Wdir' not in n and 'Etmp' not in n and 
            #     'Itmp' not in n and 'Ndir' not in n and 'Prtv' not in n 
            #     and 'Pab1' not in n and 'Pab2' not in n and 'Pab3' not in n 
            # ]
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        log.info('adding time')
        if self.test_x:
            t = df_data['Tmstamp'].apply(lambda x: x)
        else:
            t = df_data['Tmstamp'].apply(func_add_t)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(to_replace=np.nan, value=0)

        return new_df_data, raw_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]
        data = df_data.values
        raw_data = raw_df_data.values

        data = np.reshape(data, [self.capacity, -1, len(cols_data)])
        raw_data = np.reshape(raw_data, [self.capacity, -1, len(cols_data)])

        data_x = data[:, :, :]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id], columns=cols_data))
        return np.expand_dims(data_x, [0])

    def get_data(self):
        return self.data_x


if __name__ == "__main__":
    data_path = "./data"
    data = PGL4WPFDataset(data_path, filename="wtb5_10.csv")
