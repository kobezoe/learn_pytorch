import pandas as pd

v1 = ['男', '女']
v2 = ['高', '中']
i = ['1', '2']
s1 = pd.Series(v1, index=i)
s2 = pd.Series(v2, index=i)

df = pd.DataFrame({'sex': s1, 'height': s2})
print(df)
