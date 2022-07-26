# KDDCup 22 Wind Power Forecasting
## Introduction
Wind Power Forecasting (WPF) aims to accurately estimate the wind power supply of a wind farm at different time scales. 
Wind power is one of the most installed renewable energy resources in the world, and the accuracy of wind power forecasting method directly affects dispatching and operation safety of the power grid.
WPF has been widely recognized as one of the most critical issues in wind power integration and operation. 


## Data Description
Please refer to KDD Cup 2022 --> Wind Power Forecast --> Task Definition 
(https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)

Download data and place it into `./data`

## Requirements

```
pgl==2.2.3post0
paddlepaddle-gpu>=2.2.2
```

## Performance

|        | Dev Score | Max-dev Test Score |
|--------|-----------|--------------------|
| Report |   -       | 47.7               |
| Ours   | 38.93     | 46.83              |

## Prediction Visualization

During Training we visualize the prediction in devided validation and test set. See `val_vis.png` and `test_vis.png`
