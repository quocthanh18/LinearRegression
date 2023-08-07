import pandas as pd

#calculate the average of the final column
df = pd.read_csv('train.csv')
total = df.iloc[:, -1].sum()
print(total / len(df))
