import pandas as pd

datas = ["这是一条伪造的数据"] * 100

train_df = pd.DataFrame(datas,columns=["question"])

print(train_df)

train_df.to_parquet('./train.parquet')

datas = ["这是一条伪造的数据"] * 8

test = pd.DataFrame(datas,columns=["question"])


test.to_parquet('./test.parquet')