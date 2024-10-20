import pandas as pd
import numpy as np
from pprint import pprint

#Read data
df = pd.read_pickle("Data/icd_clean.pkl")[['cause','causa_icd']].sample(300_000, random_state=2011)
df.rename(columns={'causa_icd':'label'}, inplace=True) #rename column causa_icd to label

#Save data
df.to_pickle('Data/icd_clean.pkl')




