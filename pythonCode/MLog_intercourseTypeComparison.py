import math
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import nn, random, vmap
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


# for encoding

from sklearn.preprocessing import LabelEncoder


PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_model_DF.csv")

pd.set_option('display.max_columns', None)
PKD_model_DF.head




filtered_df = PKD_model_DF[PKD_model_DF["OralProtec"] != "noOral"]

total_counts = len(filtered_df)  # Total number of data points

# Create a new DataFrame with the percentages
percentage_df = filtered_df.groupby(["Oral", "OralProtec", "HIV"]).size().reset_index(name="Count")

percentage_df["Percentage"] = (percentage_df["Count"] / total_counts) * 100




sns.set(style="whitegrid")

# Create a barplot using Seaborn
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.catplot(
    data=PKD_model_DF,
    x="Oral",
    kind="count",
    col="OralProtec",
    hue="HIV",
    palette="Set1",
    height=4,  # Adjust the height as needed
    aspect=0.7,  # Adjust the aspect ratio as needed
)

# Set labels and titles
g.set_axis_labels("Oral Preferences Categories", "Count")
g.set_titles("Oral Protection: {col_name}")

# Show the plot
plt.show()

# percentages


sns.set(style="whitegrid")




# Create a barplot using Seaborn
g = sns.catplot(
    data=percentage_df,
    x="Oral",
    y="Percentage",
    kind="bar",
    col="OralProtec",
    hue="HIV",
    palette="Set1",
    height=4,  # Adjust the height as needed
    aspect=0.7,  # Adjust the aspect ratio as needed
)

# Set labels and titles
g.set_axis_labels("Oral Preferences Categories", "Percentage")
g.set_titles("Oral Protection: {col_name}")


plt.show()





# Oral
# Active, a person using their mouth
# Passive, a person at the receiving end



PKD_model_DF['Oral'].value_counts()
PKD_model_DF['OralProtec'].value_counts()
PKD_model_DF['Oral'].unique()
PKD_model_DF['OralProtec'].unique()


label_encoder = LabelEncoder()

PKD_model_DF['Oral_enc'] = label_encoder.fit_transform(PKD_model_DF['Oral'])
# (['no', 'vers', 'active', 'passive'] / ([1, 3, 0, 2])
PKD_model_DF['Oral_enc'].unique()


PKD_model_DF['OralProtec_enc'] = label_encoder.fit_transform(PKD_model_DF['OralProtec'])
# (['noOral', 'always', 'sometimes', 'never'] / ([2, 0, 3, 1])
PKD_model_DF['OralProtec_enc'].unique()


dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'Oral_enc': PKD_model_DF.Oral_enc.values, # 4 (['no', 'vers', 'active', 'passive'] / ([1, 3, 0, 2])
    'OralProtec_enc': PKD_model_DF.OralProtec_enc.values, # 4 (['noOral', 'always', 'sometimes', 'never'] / ([2, 0, 3, 1])
}

# The Model

def model(Oral_enc, OralProtec_enc, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([4]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([4]))

    logit_p = a + b[Oral_enc] + c[OralProtec_enc]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_OralSex = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_OralSex.run(random.PRNGKey(0), **dat_list)
HIV_OralSex.print_summary(0.89)


Post_OralSexModel = HIV_OralSex.get_samples()


DFPost_OralSexModel = pd.DataFrame()


for v in range(4): # Loop for 
    postDFOne = vmap(lambda k: expit(Post_OralSexModel["a"] +
                                    Post_OralSexModel["b"][:, v] + Post_OralSexModel["c"][:, k]), 0, 1)(jnp.arange(4))
    postDFOne = pd.DataFrame(postDFOne, columns=['always', 'never', 'noOral', 'sometimes']) # (['noOral', 'always', 'sometimes', 'never'] / ([2, 0, 3, 1])
    
    postDFLong = pd.melt(postDFOne)
    postDFLong['Oral_enc'] = jnp.repeat(v, len(postDFLong))
    DFPost_OralSexModel = pd.concat([DFPost_OralSexModel, postDFLong])


DFPost_OralSexModel.rename(columns={'variable': 'OralProt', 'value': 'Probability'}, inplace=True)

category_mapping_Oral = {0: 'no', 1: 'vers', 2: 'active', 3: 'passive'} # 4 (['no', 'vers', 'active', 'passive'] 
DFPost_OralSexModel['Oral_enc'] = DFPost_OralSexModel['Oral_enc'].map(category_mapping_Oral)

##
import pickle

with open('DFPost_OralSexModel', 'wb') as f: # saving pickle file
    pickle.dump(DFPost_OralSexModel, f)
##
with open('DFPost_OralSexModel', 'rb') as file:  # loading pickle saved file
    DFPost_OralSexModel = pickle.load(file)




# vis


import seaborn as sns

g = sns.FacetGrid(DFPost_OralSexModel, col='OralProt')
g.map_dataframe(sns.violinplot, x="Oral_enc", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()



# for anal intercourse now

# from this file: MLogit_anal_prot ##################

import pickle

with open('PostAnal.pkl', 'rb') as file:  # loading pickle saved file
    PostAnal = pickle.load(file)


postDF_Anals = pd.DataFrame()

for n in range(4):
    postDFOne = vmap(lambda k: expit(PostAnal["a"] + PostAnal["b"][:, k] + PostAnal["c"][:, n]), 0, 1)(jnp.arange(4))
    postDFOne = pd.DataFrame(postDFOne, columns=['always', 'never', 'noAnal', 'sometimes'])   # ['noAnal', 'always', 'sometimes', 'never'] / [2, 0, 3, 1]
    postDFLong = pd.melt(postDFOne)
    postDFLong['AnalRole'] = jnp.repeat(n, len(postDFLong))
    postDF_Anals = pd.concat([postDF_Anals, postDFLong])


sanity_check = postDF_Anals.groupby(['variable', 'AnalRole']).size().reset_index(name='Count')

postDF_Anals.rename(columns={'variable': 'AnalProtection', 'value': 'Probability'}, inplace=True)

postDF_Anals.head()


category_mapping_Anal = {0: 'no', 1: 'vers', 2: 'active', 3: 'passive'} # ['no', 'vers', 'active', 'passive'] / ([1, 3, 0, 2])
postDF_Anals['AnalRole'] = postDF_Anals['AnalRole'].map(category_mapping_Anal)



# and let's compare those two models

# DFPost_OralSexModel
# First Visualization



g1 = sns.FacetGrid(DFPost_OralSexModel, col='OralProt')
g1.map_dataframe(sns.violinplot, x="Oral_enc", y="Probability")
g1.set_axis_labels("", "Probability")
g1.set(ylim=(0, 0.1))  # Set y-axis limits

# Second Visualization
g2 = sns.FacetGrid(postDF_Anals, col='AnalProtection')
g2.map_dataframe(sns.violinplot, x="AnalRole", y="Probability")
g2.set_axis_labels("", "Probability")
g2.set(ylim=(0, 0.1))  # Set y-axis limits

# Create a 2x1 grid for subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Add the first visualization to the first subplot
g1.set_axis_labels("", "Probability", ax=axes[0])

# Add the second visualization to the second subplot
g2.set_axis_labels("", "Probability", ax=axes[1])

# Set titles for subplots
axes[0].set_title("First Visualization")
axes[1].set_title("Second Visualization")

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure with both subplots
plt.show()