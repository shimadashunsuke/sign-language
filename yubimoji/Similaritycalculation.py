import pandas as pd
import numpy as np
# CSVファイル保存先
csv_path = './point_history_add_label.csv'

df = pd.read_csv(csv_path, index_col=0)
df2 = df.query('fingernum == 13') - df.query('fingernum == 5')

v_x1 = np.array([])
for i in range(2, 35, 16):
    v_x1 = np.append(v_x1, df2.iat[0,i])


df_fing_0 = df.query('fingernum == 0')
df_fing_4 = df.query('fingernum == 4')
df_fing_8 = df.query('fingernum == 8')
df_fing_12 = df.query('fingernum == 12')
df_fing_16 = df.query('fingernum == 16')
df_fing_20 = df.query('fingernum == 20')

df_4_0 = df_fing_4 - df_fing_0
df_8_0 = df_fing_8 - df_fing_0
df_12_0 = df_fing_12 - df_fing_0
df_16_0 = df_fing_16 - df_fing_0
df_20_0 = df_fing_20 - df_fing_0

v_x2 = np.array([])

for i in range(2, 35, 16):
    v_x2 = np.append(v_x2, df_4_0.iat[0, i])

for i in range(2, 35, 16):
    v_x2 = np.append(v_x2, df_8_0.iat[0, i])

for i in range(2, 35, 16):
    v_x2 = np.append(v_x2, df_12_0.iat[0, i])

for i in range(2, 35, 16):
    v_x2 = np.append(v_x2, df_16_0.iat[0, i])

for i in range(2, 35, 16):
    v_x2 = np.append(v_x2, df_20_0.iat[0, i])

v = []
v.append(v_x1)
v.append(v_x2)