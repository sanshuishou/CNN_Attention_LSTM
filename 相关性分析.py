import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#热力图生成
data = pd.read_excel(r'C:\Users\20426\Desktop\okk.xlsx',sheet_name='Sheet1')
df = data.iloc[1:1161,1:14]
result5 = df.corr(method='pearson')
# np.save('相关性分析数据.npy',dataset)
# dataset = np.load('相关性分析数据.npy')
# df = pd.DataFrame(dataset)
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(font_scale=0.5,rc=rc)  # 设置字体大小
sns.heatmap(result5,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            linewidth=0.5,  # 设置每个单元格的距离
            linecolor='blue',  # 设置间距线的颜色
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            cmap='coolwarm_r',  # 设置热力图颜色
            )
plt.savefig("我是热力图.png",dpi=2000)#保存图片，分辨率为600
plt.ion() #显示图片
