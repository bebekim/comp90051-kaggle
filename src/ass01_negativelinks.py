import pandas as pd

df = pd.read_csv('input/sample_fe.csv')
del df['Unnamed: 0']
sample = list(df.itertuples(index=False, name=None))

# build outcome 0 for 
## source class 1
## source class 2
## ...
## source class 6

node_set = {}
for i, k in enumerate(sample):
    if i == 10:
        break
    else:
        print(f'{i} and {k}')