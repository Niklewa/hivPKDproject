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

######









PKD_model_DF['LongPartnersAmount'].unique()



# Risk model (many variab.)

PKD_model_DF['Seks_alkohol'].unique()
PKD_model_DF["Seks_alkohol"].isna().sum()
PKD_model_DF.dropna(subset=['Seks_alkohol'], inplace=True)

PKD_model_DF['Anal'].unique()


PKD_model_DF = PKD_model_DF[PKD_model_DF['Płeć'] != 'I']
label_encoder = LabelEncoder()
PKD_model_DF['Płeć_encoded'] = label_encoder.fit_transform(PKD_model_DF['Płeć'])
# M 1 / K 0
PKD_model_DF['Płeć_encoded'].unique()


label_encoder = LabelEncoder()
PKD_model_DF['Anal_enc'] = label_encoder.fit_transform(PKD_model_DF['Anal'])
# ['no', 'vers', 'active', 'passive'] / ([1, 3, 0, 2])
PKD_model_DF['Anal_enc'].unique()


PKD_model_DF = PKD_model_DF[PKD_model_DF['ShortPartnersAmount'] != 'na']

PKD_model_DF['ShortPartnersAmount'].unique()

label_encoder = LabelEncoder()
PKD_model_DF['ShortPartnersAmount_enc'] = label_encoder.fit_transform(PKD_model_DF['ShortPartnersAmount'])
# ['1-10', '11-50', 'above_51'] / ([0, 1, 2])
PKD_model_DF['ShortPartnersAmount_enc'].unique()



dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'Seks_alkohol': PKD_model_DF.Seks_alkohol.values.astype('int64'), # 2
    'Anal_enc': PKD_model_DF.Anal_enc.values, # 4
    'ShortPartnersAmount_enc': PKD_model_DF.ShortPartnersAmount_enc.values, # 3
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values, # 2

}

# Model

def model(Płeć_encoded, ShortPartnersAmount_enc, Seks_alkohol, Anal_enc, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))
    d = numpyro.sample("d", dist.Normal(0, 0.5).expand([3]))
    e = numpyro.sample("e", dist.Normal(0, 0.5).expand([4]))


    logit_p = a + b[Seks_alkohol] + c[Płeć_encoded] + d[ShortPartnersAmount_enc] + e[Anal_enc] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_RiskProfile = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_RiskProfile.run(random.PRNGKey(0), **dat_list)
HIV_RiskProfile.print_summary(0.89)



Post_HIVRiskProfile = HIV_RiskProfile.get_samples()

##
import pickle

with open('post.Post_HIVRiskProfile', 'wb') as f:
    pickle.dump(Post_HIVRiskProfile, f)
##

with open('post.Post_HIVRiskProfile', 'rb') as file:  # loading ppickle saved file
    Post_HIVRiskProfile = pickle.load(file)

DFPost_HIVRiskProfile = pd.DataFrame()

for n in range(2):  # Loop for b
    for m in range(2):  # Loop for c
        for v in range(3): # Loop for d
            postDFOne = vmap(lambda k: expit(Post_HIVRiskProfile["a"] + Post_HIVRiskProfile["b"][:, n] + Post_HIVRiskProfile["c"][:, m] +
                                            Post_HIVRiskProfile["d"][:, v] + Post_HIVRiskProfile["e"][:, k]), 0, 1)(jnp.arange(4))
            postDFOne = pd.DataFrame(postDFOne, columns=['active', 'no', 'passive', 'vers']) # ['no', 'vers', 'active', 'passive'] / ([1, 3, 0, 2])
            postDFLong = pd.melt(postDFOne)
            postDFLong['Sex_alcohol'] = jnp.repeat(n, len(postDFLong))
            postDFLong['Gender'] = jnp.repeat(m, len(postDFLong))
            postDFLong['PartnersNum'] = jnp.repeat(v, len(postDFLong))
            DFPost_HIVRiskProfile = pd.concat([DFPost_HIVRiskProfile, postDFLong])


DFPost_HIVRiskProfile.head()


sanityCheck = DFPost_HIVRiskProfile.groupby(['variable', 'Sex_alcohol', 'PartnersNum', 'Gender']).size().reset_index(name='Count')

DFPost_HIVRiskProfile.rename(columns={'variable': 'AnalPosition', 'value': 'Probability'}, inplace=True)




DFPost_HIVRiskProfile.to_csv('DFPost_HIVRiskProfile.csv', index=False)

## Vis

import seaborn as sns

import seaborn as sns
from matplotlib.lines import Line2D

sns.set(style="whitegrid")

custom_palette = ["#8D4585", "#56B4E9", '#FFC20A'] 



# FOR MALES

postDF_RiskProfileMales = DFPost_HIVRiskProfile[DFPost_HIVRiskProfile['Gender'] == 1]
postDF_RiskProfileFemales = DFPost_HIVRiskProfile[DFPost_HIVRiskProfile['Gender'] == 0]

postDF_Alco1 = postDF_RiskProfileMales[postDF_RiskProfileMales['Sex_alcohol'] == 1]
postDF_Alco0 = postDF_RiskProfileMales[postDF_RiskProfileMales['Sex_alcohol'] == 0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))  
# Plot for gender = 1
sns.violinplot(data=postDF_Alco1, x="AnalPosition", y="Probability", hue="PartnersNum", ax=axes[0], palette=custom_palette)
axes[0].set_title("Alco Sex")
axes[0].set_xlabel("")
axes[0].set_ylabel("Probability")

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[0], markersize=10, label='1-10'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[1], markersize=10, label='11-50'), # '1-10', '11-50', 'above_51'
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[2], markersize=10, label='above 51')]


legend2 = axes[0].legend(title="Number of Partners", loc="upper left", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# Plot for gender = 0
sns.violinplot(data=postDF_Alco0, x="AnalPosition", y="Probability", hue="PartnersNum", ax=axes[1], palette=custom_palette)
axes[1].set_title("No Alco Sex")
axes[1].set_xlabel("Anal Sex Position")
axes[1].set_ylabel("Probability")


lelegend2 = axes[1].legend(title="Number of Partners", loc="upper left", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# unifying y axis values
combined_data = DFPost_HIVRiskProfile["Probability"]
y_min = combined_data.min()
y_max = combined_data.max()
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

plt.suptitle("Probability of Getting HIV Infected (Males Only)")

plt.tight_layout()
plt.show()



# FOR FEMALES ##################################################


postDF_RiskProfileFemales = DFPost_HIVRiskProfile[DFPost_HIVRiskProfile['Gender'] == 0]

postDF_Alco1 = postDF_RiskProfileFemales[postDF_RiskProfileFemales['Sex_alcohol'] == 1]
postDF_Alco0 = postDF_RiskProfileFemales[postDF_RiskProfileFemales['Sex_alcohol'] == 0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))  
# Plot for gender = 1
sns.violinplot(data=postDF_Alco1, x="AnalPosition", y="Probability", hue="PartnersNum", ax=axes[0], palette=custom_palette)
axes[0].set_title("Alco Sex")
axes[0].set_xlabel("")
axes[0].set_ylabel("Probability")

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[0], markersize=10, label='1-10'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[1], markersize=10, label='11-50'), # '1-10', '11-50', 'above_51'
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[2], markersize=10, label='above 51')]


legend2 = axes[0].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# Plot for gender = 0
sns.violinplot(data=postDF_Alco0, x="AnalPosition", y="Probability", hue="PartnersNum", ax=axes[1], palette=custom_palette)
axes[1].set_title("No Alco Sex")
axes[1].set_xlabel("Anal Sex Position")
axes[1].set_ylabel("Probability")


lelegend2 = axes[1].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# unifying y axis values
combined_data = DFPost_HIVRiskProfile["Probability"]
y_min = combined_data.min()
y_max = combined_data.max()
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

plt.suptitle("Probability of Getting HIV Infected (Females Only)")

plt.tight_layout()
plt.show()

############# Profiling Model

# Anal(passive, vers) / Seks_alkohol (1.) / ShortPartnersAmount (not 1-10) (mix it with protection) /
# AnalProtec(sometimes, never) # Płeć (M) (don't take gender as causal parameter)


DF_data_RProfile = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_modelHIGHRISK_DF.csv")

DF_data_RProfile['Anal'].unique()
label_encoder = LabelEncoder()
DF_data_RProfile['Anal_enc'] = label_encoder.fit_transform(PKD_model_DF['Anal'])
# ['no', 'vers', 'passive', 'active'] / ([1, 3, 2, 0])
DF_data_RProfile['Anal_enc'].unique()


DF_data_RProfile = DF_data_RProfile[DF_data_RProfile['ShortPartnersAmount'] != 'na']

DF_data_RProfile['ShortPartnersAmount'].unique()

label_encoder = LabelEncoder()
DF_data_RProfile['ShortPartnersAmount_enc'] = label_encoder.fit_transform(DF_data_RProfile['ShortPartnersAmount'])
# ['1-10', '11-50', 'above_51'] / ([0, 1, 2])
DF_data_RProfile['ShortPartnersAmount_enc'].unique()


dat_list = {
    'HIV': DF_data_RProfile.HIV.values,  # 2
    'Seks_alkohol': DF_data_RProfile.Seks_alkohol.values.astype('int64'), # 2
    'Anal_enc': DF_data_RProfile.Anal_enc.values, # 4
    'ShortPartnersAmount_enc': DF_data_RProfile.ShortPartnersAmount_enc.values, # 3
    'Hetero_normative': DF_data_RProfile.Hetero_normative.values, # 2
}

# Model

def model(Hetero_normative, ShortPartnersAmount_enc, Seks_alkohol, Anal_enc, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))
    d = numpyro.sample("d", dist.Normal(0, 0.5).expand([3]))
    e = numpyro.sample("e", dist.Normal(0, 0.5).expand([4]))


    logit_p = a + b[Seks_alkohol] + c[Hetero_normative] + d[ShortPartnersAmount_enc] + e[Anal_enc] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_RiskProfiler = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_RiskProfiler.run(random.PRNGKey(0), **dat_list)
HIV_RiskProfiler.print_summary(0.89) 




Post_HIVRiskProfiler = HIV_RiskProfiler.get_samples()

##
import pickle

with open('post.Post_HIVRiskProfiler', 'wb') as f:
    pickle.dump(Post_HIVRiskProfiler, f)
##


DFPost_HIVRiskProfiler = pd.DataFrame()

for n in range(2):  # Loop for b
    for m in range(2):  # Loop for c
        for v in range(1,3): # Loop for d
            postDFOne = vmap(lambda k: expit(Post_HIVRiskProfiler["a"] + Post_HIVRiskProfiler["b"][:, n] + Post_HIVRiskProfiler["c"][:, m] +
                                            Post_HIVRiskProfiler["d"][:, v] + Post_HIVRiskProfiler["e"][:, k]), 0, 1)(jnp.arange(4))
            postDFOne = pd.DataFrame(postDFOne, columns=['passive', 'no', 'active', 'vers']) # ['no', 'vers', 'passive', 'active'] / ([1, 3, 0, 2])
            postDFLong = pd.melt(postDFOne)
            postDFLong['Sex_alcohol'] = jnp.repeat(n, len(postDFLong))
            postDFLong['HeteroNorm'] = jnp.repeat(m, len(postDFLong))
            postDFLong['PartnersNum'] = jnp.repeat(v, len(postDFLong))
            DFPost_HIVRiskProfiler = pd.concat([DFPost_HIVRiskProfiler, postDFLong])


DFPost_HIVRiskProfiler = DFPost_HIVRiskProfiler.rename(columns={'variable': 'AnalPosition', 'value': 'Probability'}, inplace=True)
DFPost_HIVRiskProfiler.head()



