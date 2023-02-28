import pandas as pd
import numpy as np
#其实就是数据预处理,不用管,经供参考
data = pd.read_excel(r'C:\Users\20426\Desktop\okk.xlsx',sheet_name='Sheet1')

dataset = data.iloc[1:1261,:14].values
# np.save('10_12.npy',dataset)
# dataset = np.load('10_12.npy')
dataset = np.nan_to_num(dataset)
a = []
b = []
for i in range(len(dataset)):
    for j in range(len(dataset[i])):#0-1
        aa = round(dataset[i][j]/30,2)
        for k in range(30):
            if j == 0:
                a.append(aa)
            else:
                b.append(aa)
c = []
for i in range(len(a)):
    kk = []
    kk.append(a[i])
    kk.append(b[i])
    c.append(kk)

c = pd.DataFrame(c)
c.to_excel('okk.xlsx')
