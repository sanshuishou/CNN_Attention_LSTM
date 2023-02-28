import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import models.cnn_attention_lstm as mv
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
#此处为预测程序
a = np.load('train_x.npy')
b = np.load('test_x.npy')
train_y_set = np.load('train_y.npy')
test_y_set = np.load('test_y.npy')
# 归一化
sc = MinMaxScaler(feature_range=(0 , 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(a)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
y_training_set_scaled = sc.fit_transform(train_y_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化

test_set_scaled = sc.fit_transform(b)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
y_test_set_scaled = sc.fit_transform(test_y_set)  # 利用训练集的属性对测试集进行归一化


x_train = []
y_train = []

x_test = []
y_test = []

# 利用for循环，遍历整个训练集，提取训练集中连续30天的收盘价作为输入特征x_train，第30天的数据作为标签，for循环共构建1135-235-60=840组数据。
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, :])
    y_train.append(y_training_set_scaled[i, :])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
y_train = np.reshape(y_train,(y_train.shape[0],))

# print(x_train.shape)
# print(y_train.shape)
# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。

# 利用for循环，遍历整个测试集，提取测试集中连续30天的收盘价作为输入特征x_train，第31天的数据作为标签，for循环共构建235-30=200组数据。
for i in range(60, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 60:i, :])
    y_test.append(y_test_set_scaled[i, :])
# 测试集变array并reshape为符合LSTM输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
y_test = np.reshape(y_test,(y_test.shape[0],)) #标签为一维数组


model = mv.conv_lstm(60,6,128) #attention_lstm(时间步长，特征数量，lstm神经元数量)


model.load_weights('./checkpoint_/my_modelv2.ckpt')

################## predict ######################
# 测试集输入模型进行预测
predicted_stock_price_t = model.predict(x_train)

# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price_t = sc.inverse_transform(predicted_stock_price_t)

# 对真实数据还原---从（0，1）反归一化到原始范围
y_train = tf.reshape(y_train,(-1,1))
real_stock_price_t = sc.inverse_transform(y_train)
print(real_stock_price_t.shape)

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price_t, color='red', label='Maize Close Price')
plt.plot(predicted_stock_price_t, color='blue', label='Predicted Maize Close Price')
plt.title('Maize Close Price traindataset Prediction')
plt.xlabel('Time')
plt.ylabel('Maize Close Price')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price_t, real_stock_price_t)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price_t, real_stock_price_t))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price_t, real_stock_price_t)
mape = mean_absolute_percentage_error(predicted_stock_price_t, real_stock_price_t)
print('MSE: %.6f' % mse)
print('RMSE: %.6f' % rmse)
print('MAE: %.6f' % mae)
print('MAPE: %.6f' % mape)