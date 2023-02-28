import tensorflow as tf
import pandas as pd
import numpy as np
import models.cnn_attention_lstm as mv
from sklearn.preprocessing import MinMaxScaler
#模型训练程序

# dataframe = pd.read_excel('原数据/相关性分析用表.xlsx','Sheet1')
# train_set = dataframe.iloc[375:1876,1:2].values
# test_set = dataframe.iloc[1:375,1:2].values
# tz_train_set = dataframe.iloc[375:1876,1:7].values
# tz_test_set = dataframe.iloc[1:375,1:7].values
#
# train_set = np.nan_to_num(train_set,nan=2844)
# test_set = np.nan_to_num(test_set,nan=2844)
# tz_train_set = np.nan_to_num(tz_train_set,nan=2923)
# tz_test_set = np.nan_to_num(tz_test_set,nan=2923)
#
# np.save('train_x.npy',tz_train_set)
# np.save('test_x.npy',tz_test_set)
# np.save('train_y.npy',train_set)
# np.save('test_y.npy',test_set)

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

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(1e-4))

histroy = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3000,batch_size=128,callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoint_/my_modelv2.ckpt',save_weights_only=True,save_best_only=True,verbose=1)])
