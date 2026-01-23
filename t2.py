import pandas as pd
import ast

# 读取文件
df = pd.read_csv(r'D:\codes\Uplink_Compute\source\video_1\view.csv')

for i in range(len(df)):
    index_val = df.loc[i, 'index']
    faces_val = df.loc[i, 'high_frequency_faces']
    print(f"第 {index_val} 行: {faces_val}")