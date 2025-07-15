import pandas as pd
import numpy as np 

df = pd.read_csv("creditcard.csv")

data_dict = df.to_dict()

print(data_dict.keys())

