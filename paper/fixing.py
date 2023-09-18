import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




PKD_model_DF = pd.read_csv("C:/Users/nikod/Documents/RProjects/hivProject/PKD_model_DF.csv")





category_counts = PKD_model_DF.groupby(["Płeć", "Hetero_normative", "HIV"]).size().reset_index(name='Count')

category_counts = category_counts.loc[category_counts['Płeć'] != 'I']
category_counts = category_counts.loc[category_counts['HIV'] != 0]

category_counts['Percentage'] = category_counts['Count'] / sum(category_counts['Count']) 



plt.figure(figsize=(8, 6))
sns.barplot(data=category_counts, x='Płeć', y='Percentage', hue='Hetero_normative')

plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Percentage of HIV = 1 by Gender and Sexual Identity')

# Show the plot
plt.show()