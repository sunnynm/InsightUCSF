import time
import datetime
start_time = time.time()

import pandas as pd
df = pd.read_csv("./MutliKeywordDistFULL.csv") 
alpha = df.loc[(df['ManRefVal'] >= .66)]
print(len(set(df.Document.values)))
print(len(set(alpha.Document.values)))