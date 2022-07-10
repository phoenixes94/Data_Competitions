# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Some useful metrics
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MetricsError(Exception):
    """
    Desc:
        Customize the Exception
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


def is_valid_prediction(prediction, idx=None):
    """
    Desc:
        Check if the prediction is valid
    Args:
        prediction:
        idx:
    Returns:
        A boolean value
    """
    try:
        if prediction.ndim > 1:
            nan_prediction = pd.isna(prediction).any(axis=1)
            if nan_prediction.any():
                if idx is None:
                    msg = "NaN in predicted values!"
                else:
                    msg = "NaN in predicted values ({}th prediction)!".format(idx)
                raise MetricsError(msg)
        if prediction.size == 0:
            if idx is None:
                msg = "Empty prediction!"
            else:
                msg = "Empty predicted values ({}th prediction)! ".format(idx)
            raise MetricsError(msg)
    except ValueError as e:
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        else:
            raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return True


def mae(pred, gt, run_id=0):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        MAE value
    """
    _mae = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt, run_id=0):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        MSE value
    """
    _mse = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt, run_id=0):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        RMSE value
    """
    _mse = mse(pred, gt, run_id=run_id)
    if _mse < 0:
        return -1
    return np.sqrt(_mse)


def regressor_scores(prediction, gt, idx=0):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        prediction:
        gt: ground truth vector
        idx:
    Returns:
        A tuple of metrics
    """
    _mae = mae(prediction, gt, run_id=idx)
    _rmse = rmse(prediction, gt, run_id=idx)
    return _mae, _rmse


def turbine_scores(pred, gt, raw_data, examine_len, idx=0):
    """
    Desc:
        Calculate the MAE and RMSE of one turbine
    Args:
        pred: prediction for one turbine
        gt: ground truth
        raw_data: the DataFrame of one wind turbine
        examine_len:
        idx:
    Returns:
        The averaged MAE and RMSE
    """
    nan_cond = pd.isna(raw_data).any(axis=1)
    invalid_cond = (raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720))
    indices = np.where(~nan_cond & ~invalid_cond)
    prediction = pred[indices]
    targets = gt[indices]
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    _mae, _rmse = -1, -1
    if np.any(prediction) and np.any(targets):
        _mae, _rmse = regressor_scores(prediction[-examine_len:] / 1000, targets[-examine_len:] / 1000, idx=idx)
    return _mae, _rmse


def check_zero_prediction(prediction, idx=None):
    """
    Desc:
       If zero prediction, return -1
    Args:
        prediction:
        idx:
    Returns:
        An integer indicating status
    """
    if not np.any(prediction):
        if idx is None:
            msg = "Zero prediction!"
        else:
            msg = "Zero predicted values ({}th prediction)! ".format(idx)
        print(msg)
        return -1
    return 0


def is_zero_prediction(predictions, identifier, settings):
    """
    Desc:
        Check if zero prediction for all turbines in a wind farm
    Args:
        predictions:
        identifier:
        settings:
    Returns:
        False if otherwise
    """
    wind_farm_statuses = []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        status = check_zero_prediction(prediction, idx=identifier)
        wind_farm_statuses.append(status)
    statuses = np.array(wind_farm_statuses)
    non_zero_predictions = statuses[statuses == 0]
    non_zero_ratio = non_zero_predictions.size / settings["capacity"]
    if non_zero_ratio < settings["min_non_zero_ratio"]:
        msg = "{:.2f}% turbines with zero predicted values " \
              "(in {}th prediction)!".format((1 - non_zero_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def check_identical_prediction(prediction, min_std=0.1, min_distinct_ratio=0.1, idx=None):
    """
    Desc:
        Check if the prediction is with the same values
    Args:
        prediction:
        min_std:
        min_distinct_ratio:
        idx:
    Returns:
        An integer indicating the status
    """
    try:
        if np.min(prediction) == np.max(prediction):
            if idx is None:
                msg = "All predicted values are the same as {:.4f}!".format(np.min(prediction))
            else:
                msg = "All predicted values are same as {:.4f} ({}th prediction)!".format(np.min(prediction), idx)
            print(msg)
            return -1
        if np.std(prediction) <= min_std:
            prediction = np.ravel(prediction)
            distinct_prediction = set(prediction)
            distinct_ratio = len(distinct_prediction) / np.size(prediction)
            samples = list(distinct_prediction)[:3]
            samples = ",".join("{:.5f}".format(s) for s in samples)
            if distinct_ratio < min_distinct_ratio:
                if idx is None:
                    msg = "{:.2f}% of predicted values are same! Some predicted values are: " \
                          "{},...".format((1 - distinct_ratio) * 100, samples)
                else:
                    msg = "{:.2f}% of predicted values are same ({}th run)! " \
                          "Some predicted values are:" \
                          "{},...".format((1 - distinct_ratio) * 100, idx, samples)
                print(msg)
                return -1
    except ValueError as e:
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        else:
            raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return 0


def is_identical_prediction(predictions, identifier, settings):
    """
    Desc:
        Check if the predicted values are identical for all turbines
    Args:
        predictions:
        identifier:
        settings:
    Returns:
        False
    """
    farm_check_statuses = []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        status = check_identical_prediction(prediction, min_distinct_ratio=settings["min_distinct_ratio"],
                                            idx=identifier)
        farm_check_statuses.append(status)
    statuses = np.array(farm_check_statuses)
    variational_predictions = statuses[statuses == 0]
    variation_ratio = variational_predictions.size / settings["capacity"]
    if variation_ratio < settings["min_distinct_ratio"]:
        msg = "{:.2f}% turbines with (almost) identical predicted values " \
              "({}th prediction)!".format((1 - variation_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def regressor_detailed_scores(predictions, gts, raw_df_lst, settings):
    """
    Desc:
        Some common metrics
    Args:
        predictions:
        gts: ground truth vector
        raw_df_lst:
        settings:
    Returns:
        A tuple of metrics
    """
    path_to_test_x = settings["path_to_test_x"]
    tokens = os.path.split(path_to_test_x)
    identifier = int(tokens[-1][:-6]) - 1
    all_mae, all_rmse = [], []
    all_latest_mae, all_latest_rmse = [], []
    # add
    one_day_all_mae, one_day_all_rmse = [], []
    two_day_all_mae, two_day_all_rmse = [], []
    step = 144
    if not is_identical_prediction(predictions, identifier, settings) and \
            not is_zero_prediction(predictions, identifier, settings):
        pass
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        if not is_valid_prediction(prediction, idx=identifier):
            continue
        gt = gts[i]
        raw_df = raw_df_lst[i]
        _mae, _rmse = turbine_scores(prediction, gt, raw_df, settings["output_len"], idx=identifier)
        if _mae != _mae or _rmse != _rmse:  # In case NaN is encountered
            continue
        if -1 == _mae or -1 == _rmse:       # In case the target is empty after filtering out the abnormal values
            continue
        all_mae.append(_mae)
        all_rmse.append(_rmse)
        latest_mae, latest_rmse = turbine_scores(prediction, gt, raw_df, settings["day_len"], idx=identifier)
        all_latest_mae.append(latest_mae)
        all_latest_rmse.append(latest_rmse)
        # add
        one_mae, one_rmse = turbine_scores(prediction[:step,:], gt[:step,:], raw_df.iloc[:step,:], step)
        one_day_all_mae.append(one_mae)
        one_day_all_rmse.append(one_rmse)
        two_mae, two_rmse = turbine_scores(prediction[step:, :], gt[step:, :], raw_df.iloc[step:, :], step)
        two_day_all_mae.append(two_mae)
        two_day_all_rmse.append(two_rmse)

        # ============For 288 points changing trends============
    #     turb_mae, turb_rmse = [], []
    #     for i in range(144):
    #         i_step = i * 2
    #         i_mae, i_rmse = turbine_scores(prediction[i_step:i_step+2, :], gt[i_step:i_step+2, :], raw_df.iloc[i_step:i_step+2, :], 2)
    #         if i_mae == -1:
    #             i_mae, i_rmse = 0, 0
    #         turb_mae.append(i_mae)
    #         turb_rmse.append(i_rmse)

    #     all_turb_mae.append(turb_mae)
    #     all_turb_rmse.append(turb_rmse)
    # total_turb_mae = np.array(all_turb_mae).sum(axis=0)
    # total_turb_rmse = np.array(all_turb_rmse).sum(axis=0)
    # total_turb_mae_rmse = []
    # for i in range(len(total_turb_mae)):
    #     total_turb_mae_rmse.append((total_turb_mae[i] + total_turb_rmse[i]) / 2)
    # visualize_prediction(total_turb_mae, total_turb_rmse, total_turb_mae_rmse,  tag='288')
    # ============For 288 points changing trends============

    total_mae = np.array(all_mae).sum()
    total_rmse = np.array(all_rmse).sum()
    # add
    total_one_mae = np.array(one_day_all_mae).sum()
    total_one_rmse = np.array(one_day_all_rmse).sum()
    total_two_mae = np.array(two_day_all_mae).sum()
    total_two_rmse = np.array(two_day_all_rmse).sum()

    print(f"First day: {(total_one_rmse + total_one_mae) / 2}, \
            Second day: {(total_two_mae + total_two_rmse) / 2}" )

    if total_mae < 0 or total_rmse < 0:
        raise MetricsError("{}th prediction: summed MAE ({:.2f}) or RMSE ({:.2f}) is negative, "
                           "which indicates too many invalid values "
                           "in the prediction! ".format(identifier, total_mae, total_rmse))
    if len(all_mae) == 0 or len(all_rmse) == 0 or total_mae == 0 or total_rmse == 0:
        raise MetricsError("No valid MAE or RMSE for "
                           "all of the turbines in {}th prediction! ".format(identifier))
    total_latest_mae = np.array(all_latest_mae).sum()
    total_latest_rmse = np.array(all_latest_rmse).sum()

    # visual
    # all_mae_rmse = []
    # for i in range(len(all_mae)):
    #     all_mae_rmse.append((all_mae[i] + all_rmse[i]) / 2)
    # visualize_prediction(all_mae, all_rmse, all_mae_rmse)
    return total_mae, total_rmse, total_latest_mae, total_latest_rmse


def visualize_prediction(all_mae, all_rmse, all_mae_rmse, tag='metric'):
    """
    Desc:
        Visual metrics
    Args:
        all_mae:
        all_rmse: 
        all_mae_rmse:
    Returns:
        Plot
    """
    plt.figure()
    for i in range(1, 2):
        ax = plt.subplot(1, 1, i)
        ax.plot(all_mae, label="mae")
        ax.plot(all_rmse, label="rmse")
        ax.plot(all_mae_rmse, label="mae_rmse")
        ax.legend()
    plt.savefig(tag + "_vis.png")
    plt.close()