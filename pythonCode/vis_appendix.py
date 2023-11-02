
# for the purpose of better data presentation I will create 
# viusalizations in a notion of EDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("C:/Users/nikod/Documents/PythonProjects/hivPKDproject/dataBits/PKDjoint.csv")


# Histogram for age

ageDF = df['Wiek']
ageDF = ageDF.dropna()



with open('dataBits/ageDF.pkl', 'wb') as f:
    pickle.dump(ageDF, f)


plt.hist(ageDF, bins=20, edgecolor='k', alpha=0.7)

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution Histogram')

plt.show()

lower_bound = np.percentile(ageDF, 2.5)  
upper_bound = np.percentile(ageDF, 97.5) 

print(f"The 95% confidence interval is ({lower_bound}, {upper_bound})")


# Causes

groupedCauses = pd.read_csv('C:/Users/nikod/Documents/PythonProjects/hivPKDproject/dataBits/groupedCausesPKD.csv')

translation_dict = {
    "IDU": "IDU",
    "IDU+biseks.": "IDU+bisexual",
    "IDU+hetero": "IDU+heterosexual",
    "IDU+homo": "IDU+homosexual",
    "Zabieg medyczny": "Medical procedure",
    "ciąża": "Pregnancy",
    "ciąża u partnerki": "Partner's pregnancy",
    "inne": "Other",
    "k. biseksualne": "Bisexual contacts",
    "k. heteroseksualne": "Heterosexual contacts",
    "k. homoseksualne": "Homosexual contacts",
    "krew": "Blood",
    "krew+kontakty seksualne": "Blood and sexual contacts",
    "namowa partnera": "Partner's persuasion",
    "naruszenie skóry lub błony": "Skin or mucous membrane damage",
    "objawy osłabionej odporności": "Symptoms of weakened immunity",
    "początek nowego związku": "New relationship",
    "seks MSM": "MSM sex",
    "seks WSW": "WSW sex",
    "sex worker": "sex worker",
    "skierowanie przez lekarza": "Doctor's recommendation",
    "uszkodzenie prezerwatywy": "Condom damage",
    "wynik + partnera": "Partner's positive HIV test",
    "życzenie klienta (brak ryzyka)": "Client's wish (no risk)"
}

# Replace Polish names with English names in the dataset
groupedCauses['Powód'] = groupedCauses['Powód'].map(translation_dict)

groupedCauses = groupedCauses.rename(columns={'Powód': 'causes'})




# Sort the DataFrame by 'positive' in descending order and then by 'count' in ascending order
groupedCauses_sorted = groupedCauses.sort_values(by=['positive', 'count'], ascending=[False, True])

# Print the sorted DataFrame
print(groupedCauses_sorted)


from plotnine import ggplot, geom_col, aes, geom_text, theme_tufte, coord_flip, labs


groupedCauses['percentage'] = round(groupedCauses['percentage'], 1)

import plotly.express as px

# Sort the DataFrame by 'positive' in descending order and then by 'count' in ascending order
groupedCauses_sorted = groupedCauses.sort_values(by=['positive', 'count'], ascending=[False, True])

# Create a bar chart with the sorted DataFrame
fig = px.bar(groupedCauses_sorted, x='count', y='causes', orientation='h', color='positive',
             hover_name='causes', hover_data={'count': True, 'percentage': True})

# Show the chart
fig.show()


# R code

# names to english first 

# ggplot(smalll2, aes(x= reorder(Powód, count), y= count, fill= as.factor(positive)))+
#   geom_col()+
#   coord_flip()+ 
#   geom_text(data = subset(smalll2, positive  == 1),
#                            aes(label = paste0(round(percentage, 1), "%")), hjust = -0.2)+
#   theme_tufte()+ labs(fill = "Pozytywny:", title = "Powód wykonania badania",
#                       subtitle = "PKD, dane sumaryczne", y = "", x = "Powód")




