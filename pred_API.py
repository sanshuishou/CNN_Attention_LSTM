import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import models.cnn_attention_lstm as mv
#此处为预测程序连接前端的接口

def day_60_data_pred(data_x_set):
    """
    这是一个预测函数
    :param data_x_set:连续60天的数据集
    :return: 预测后具有六个特征的时间步
    """
    sc = MinMaxScaler(feature_range=(0 , 1))
    training_set_scaled = sc.fit_transform(data_x_set)
    x_train = tf.expand_dims(training_set_scaled,0)

    model = mv.conv_lstm(60,6,128) #attention_lstm(时间步长，特征数量，lstm神经元数量)

    model.load_weights('./checkpoint_/my_modelv2.ckpt')
    ################## predict ######################
    # 测试集输入模型进行预测
    predicted_stock_price_t = model.predict(x_train)
    c = list(predicted_stock_price_t[0])
    for i in range(5):
        c.append(np.random.uniform(0,1,1))
    c = np.expand_dims(c,0)
    predicted_stock_price_t = sc.inverse_transform(c)
    return predicted_stock_price_t

def next_nday_pred(dataset,n,method):
    """
    这是一个连续预测函数
    :param dataset: 连续前60天时间步且每个时间步6个特征的数据
    :param n: 需要连续预测的天数
    :param method: 预测方法,是一个函数
    :return: 为n天玉米主力收盘价的预测值列表
    """
    pred_day = []
    for i in range(n):
        dataset = dataset[i:, :]
        c = method(dataset)
        pred_day.append(c[0])
        dataset = list(dataset)
        for j in pred_day:
            dataset.append(j)
        dataset = np.array(dataset)
    pred_nday = []
    for k in pred_day:
        pred_nday.append(k[0])
    return np.array(pred_nday)
# if __name__ == '__main__':
#     # b = pd.read_excel(r'原数据/相关性分析用表.xlsx','Sheet1')
#     # b = b.iloc[1:61,1:7].values
#     # np.save('pred.npy',b)
#     b = np.load('pred.npy')
#     a = next_nday_pred(b,7,day_60_data_pred)
#     for i in range(len(a)):
#         print(f"未来七天的玉米主力收盘价预测第{i+1}天为:",round(a[i],2))