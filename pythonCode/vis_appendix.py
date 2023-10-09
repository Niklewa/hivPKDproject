
# for the purpose of better data presentation I will create 
# viusalizations in a notion of EDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/nikod/Documents/PythonProjects/hivPKDproject/dataBits/PKDjoint.csv")


# Histogram for age

ageDF = df['Wiek']
ageDF = ageDF.dropna()



plt.hist(ageDF, bins=20, edgecolor='k', alpha=0.7)

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution Histogram')

plt.show()

lower_bound = np.percentile(ageDF, 2.5)  
upper_bound = np.percentile(ageDF, 97.5) 

print(f"The 95% confidence interval is ({lower_bound}, {upper_bound})")


# Causes

