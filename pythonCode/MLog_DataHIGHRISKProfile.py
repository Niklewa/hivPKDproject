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




PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_modelHIGHRISK_DF.csv")

pd.set_option('display.max_columns', None)
PKD_model_DF.head

######


dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'Hetero_normative': PKD_model_DF.Hetero_normative.values, # 2
    'HighRiskHIV': PKD_model_DF.HighRiskHIV.values, # 2
}

# Model

def model(Hetero_normative, HighRiskHIV, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))


    logit_p = a + b[Hetero_normative] + c[HighRiskHIV] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_HIGHRISKVar = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_HIGHRISKVar.run(random.PRNGKey(0), **dat_list)
HIV_HIGHRISKVar.print_summary(0.89)



Post_HIV_HIGHRISKVar = HIV_HIGHRISKVar.get_samples()


##
import pickle

with open('post.Post_HIV_HIGHRISKVar', 'wb') as f:
    pickle.dump(Post_HIV_HIGHRISKVar, f)
##


DFPost_HIV_HIGHRISKVar = pd.DataFrame()


for v in range(2): # Loop for d
    postDFOne = vmap(lambda k: expit(Post_HIV_HIGHRISKVar["a"] + Post_HIV_HIGHRISKVar["b"][:, v] +
                                      Post_HIV_HIGHRISKVar["c"][:, k]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=[0, 1]) 
    postDFLong = pd.melt(postDFOne)
    postDFLong['HeteroNorm'] = jnp.repeat(v, len(postDFLong))
    DFPost_HIV_HIGHRISKVar = pd.concat([DFPost_HIV_HIGHRISKVar, postDFLong])


DFPost_HIV_HIGHRISKVar.head()


DFPost_HIV_HIGHRISKVar.groupby(['variable', 'HeteroNorm']).size().reset_index(name='Count')

DFPost_HIV_HIGHRISKVar.rename(columns={'variable': 'HighRiskHIV', 'value': 'Probability'}, inplace=True)


import seaborn as sns

g = sns.FacetGrid(DFPost_HIV_HIGHRISKVar, col='HeteroNorm')
g.map_dataframe(sns.violinplot, x="HighRiskHIV", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()

###################### Now ONLY one predictor


dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'HighRiskHIV': PKD_model_DF.HighRiskHIV.values, # 2
}

# Model

def model(HighRiskHIV, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))


    logit_p = a + b[HighRiskHIV] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_HIGHRISKVarOne = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_HIGHRISKVarOne.run(random.PRNGKey(0), **dat_list)
HIV_HIGHRISKVarOne.print_summary(0.89)



Post_HIV_HIGHRISKVarOne = HIV_HIGHRISKVarOne.get_samples()


##
import pickle

with open('post.Post_HIV_HIGHRISKVarOne', 'wb') as f:
    pickle.dump(Post_HIV_HIGHRISKVarOne, f)
##

with open('post.Post_HIV_HIGHRISKVarOne', 'rb') as f:
    Post_HIV_HIGHRISKVarOne = pickle.load(f)




postDFOne = vmap(lambda k: expit(Post_HIV_HIGHRISKVarOne["a"] + Post_HIV_HIGHRISKVarOne["b"][:, k]), 0, 1)(jnp.arange(2))
postDFOne = pd.DataFrame(postDFOne, columns=[0, 1]) 
DFPost_HIV_HIGHRISKVarOne = pd.melt(postDFOne)

DFPost_HIV_HIGHRISKVarOne.head()
DFPost_HIV_HIGHRISKVarOne.rename(columns={'variable': 'HighRiskHIV', 'value': 'Probability'}, inplace=True)


import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.violinplot(x="HighRiskHIV", y="Probability", data=DFPost_HIV_HIGHRISKVarOne)

sns.stripplot(x="HighRiskHIV", y="Probability", data=DFPost_HIV_HIGHRISKVarOne,
               jitter=True, color="black", alpha=0.1)


plt.xlabel("High Risk Profile")
plt.ylabel("Probability")
plt.title("Probability of Getting Infected")

plt.ylim(0, 0.10)


plt.show()


############################################################################# No gender HIGH RISK


dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'HighRiskHIVNoGen': PKD_model_DF.HighRiskHIVNoGen.values, # 2
}



def model(HighRiskHIVNoGen, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    logit_p = a + b[HighRiskHIVNoGen] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIV_HIGHRISKGenoOne = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_HIGHRISKGenoOne.run(random.PRNGKey(0), **dat_list)
HIV_HIGHRISKGenoOne.print_summary(0.89)





Post_HIV_HIGHRISKGenoOne = HIV_HIGHRISKGenoOne.get_samples()

##
import pickle

with open('post.Post_HIV_HIGHRISKGenoOne', 'wb') as f:
    pickle.dump(Post_HIV_HIGHRISKGenoOne, f)
##


with open('post.Post_HIV_HIGHRISKGenoOne', 'rb') as f:
    Post_HIV_HIGHRISKGenoOne = pickle.load(f)

postDFOne = vmap(lambda k: expit(Post_HIV_HIGHRISKGenoOne["a"] + Post_HIV_HIGHRISKGenoOne["b"][:, k]), 0, 1)(jnp.arange(2))
postDFOne = pd.DataFrame(postDFOne, columns=[0, 1]) 
DFPost_Post_HIV_HIGHRISKGenoOne = pd.melt(postDFOne)

DFPost_Post_HIV_HIGHRISKGenoOne.head()
DFPost_Post_HIV_HIGHRISKGenoOne.rename(columns={'variable': 'HighRiskHIVnoGender', 'value': 'Probability'}, inplace=True)

# VIS

import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.violinplot(x="HighRiskHIVnoGender", y="Probability", data=DFPost_Post_HIV_HIGHRISKGenoOne)

sns.stripplot(x="HighRiskHIVnoGender", y="Probability", data=DFPost_Post_HIV_HIGHRISKGenoOne,
               jitter=True, color="black", alpha=0.1)


plt.xlabel("High Risk Profile")
plt.ylabel("Probability")
plt.title("Probability of Getting Infected")

plt.ylim(0, 0.10)


plt.show()


# Model comparison WAIC

posterior_model1 = az.from_numpyro(HIV_HIGHRISKGenoOne)
posterior_model2 = az.from_numpyro(HIV_HIGHRISKVarOne)

waic_results = az.compare({"model1": posterior_model1, "model2": posterior_model2}, ic="waic")



# Let's try to predict being in this profile ###########################################################

# gender
# homonormative
# Not asking about HIV!!!
PKD_model_DF.head()
PKD_model_DF["Płeć"].unique()


label_encoder = LabelEncoder()
PKD_model_DF['Płeć_encoded'] = label_encoder.fit_transform(PKD_model_DF['Płeć'])
# M 1 / K 0
PKD_model_DF['Płeć_encoded'].unique()

result = PKD_model_DF.groupby(['Płeć_encoded', 'HighRiskHIVNoGen', 'Hetero_normative']).size().reset_index(name='Count')

total_counts = result.groupby('Płeć_encoded')['Count'].transform('sum')
result['Proportion'] = result['Count'] / total_counts                                           # VIS OF THAT!!!!

import seaborn as sns
import matplotlib.pyplot as plt

hetero_labels = {0: 'Non-Hetero', 1: 'Hetero'}
result['Hetero_normative_label'] = result['Hetero_normative'].map(hetero_labels)

colors = sns.color_palette("Set2", n_colors=len(result['HighRiskHIVNoGen'].unique()))

g = sns.FacetGrid(result, col='Płeć_encoded', height=5)
g.map_dataframe(sns.barplot, x='Hetero_normative_label', y='Proportion', hue='HighRiskHIVNoGen', dodge=True, palette=colors)
g.set_axis_labels("", "Proportion")
g.add_legend(title='HighRiskHIVNoGen')
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Proportions of HighRiskHIVNoGen (Category 1) by Gender and Hetero_normative')

plt.show()



dat_list = {
    'HighRiskHIVNoGen': PKD_model_DF.HighRiskHIVNoGen.values, # 2
    'Hetero_normative': PKD_model_DF.Hetero_normative.values, # 2
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values, # 2
}


def model(Hetero_normative, Płeć_encoded, HighRiskHIVNoGen=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))

    logit_p = a + b[Hetero_normative]  + c[Płeć_encoded]

    numpyro.sample("HighRiskHIVNoGen", dist.Binomial(logits=logit_p), obs=HighRiskHIVNoGen)
                                                                
RISK_NormGender = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
RISK_NormGender.run(random.PRNGKey(0), **dat_list)
RISK_NormGender.print_summary(0.89)


Post_RISK_NormGender = RISK_NormGender.get_samples()

DFPost_Post_RISK_NormGender = pd.DataFrame()

for v in range(2):
    postDFOne = vmap(lambda k: expit(Post_RISK_NormGender["a"] + Post_RISK_NormGender["c"][:, v] +
                                      Post_RISK_NormGender["b"][:, k]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=['not hetero', 'hetero']) 
    postDFLong = pd.melt(postDFOne)
    postDFLong['Gender'] = jnp.repeat(v, len(postDFLong)) 
    DFPost_Post_RISK_NormGender = pd.concat([DFPost_Post_RISK_NormGender, postDFLong])
    
    



DFPost_Post_RISK_NormGender.head()

DFPost_Post_RISK_NormGender.rename(columns={'variable': 'HeteroNorm', 'value': 'Probability'}, inplace=True)

##
import pickle

with open('DFPost_Post_RISK_NormGender.pkl', 'wb') as f:
    pickle.dump(DFPost_Post_RISK_NormGender, f)
##



import seaborn as sns

g = sns.FacetGrid(DFPost_Post_RISK_NormGender, col='Gender')
g.map_dataframe(sns.violinplot, x="HeteroNorm", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()

############# Now only hgihRisk with gender

dat_list = {
    'HighRiskHIVNoGen': PKD_model_DF.HighRiskHIVNoGen.values, # 2
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values, # 2
}



def model( Płeć_encoded, HighRiskHIVNoGen=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))

    logit_p = a   + b[Płeć_encoded]

    numpyro.sample("HighRiskHIVNoGen", dist.Binomial(logits=logit_p), obs=HighRiskHIVNoGen)
                                                                
RISK_NormGendernoHetero = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
RISK_NormGendernoHetero.run(random.PRNGKey(0), **dat_list)
RISK_NormGendernoHetero.print_summary(0.89)


Post_RISK_NormGendernoHetero = RISK_NormGendernoHetero.get_samples()




postDFOne = vmap(lambda k: expit(Post_RISK_NormGendernoHetero["a"] + 
                                    Post_RISK_NormGendernoHetero["b"][:, k]), 0, 1)(jnp.arange(2))
postDFOne = pd.DataFrame(postDFOne, columns=['0', '1']) 
DFPost_Post_RISK_NormGendernoHetero = pd.melt(postDFOne)



DFPost_Post_RISK_NormGendernoHetero.head()

DFPost_Post_RISK_NormGendernoHetero.rename(columns={'variable': 'Gender', 'value': 'Probability'}, inplace=True)


##
import pickle

with open('DFPost_Post_RISK_NormGendernoHetero.pkl', 'wb') as f:
    pickle.dump(DFPost_Post_RISK_NormGendernoHetero, f)
##




import seaborn as sns

sns.violinplot(DFPost_Post_RISK_NormGendernoHetero, x="Gender", y="Probability")
plt.xlabel("Gender")
plt.title('Probability of HighRiskProfile')
plt.ylabel("Probability")

plt.show()


############ Only heteroNorm

dat_list = {
    'HighRiskHIVNoGen': PKD_model_DF.HighRiskHIVNoGen.values, # 2
    'Hetero_normative': PKD_model_DF.Hetero_normative.values, # 2
}



def model( Hetero_normative, HighRiskHIVNoGen=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))

    logit_p = a   + b[Hetero_normative]

    numpyro.sample("HighRiskHIVNoGen", dist.Binomial(logits=logit_p), obs=HighRiskHIVNoGen)
                                                                
RISK_Hetero = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
RISK_Hetero.run(random.PRNGKey(0), **dat_list)
RISK_Hetero.print_summary(0.89)


POST_RISK_Hetero = RISK_Hetero.get_samples()




postDFOne = vmap(lambda k: expit(POST_RISK_Hetero["a"] + 
                                    POST_RISK_Hetero["b"][:, k]), 0, 1)(jnp.arange(2))
postDFOne = pd.DataFrame(postDFOne, columns=['0', '1']) 
DFPost_POST_RISK_Hetero= pd.melt(postDFOne)



DFPost_POST_RISK_Hetero.head()

DFPost_POST_RISK_Hetero.rename(columns={'variable': 'HeteroNorm', 'value': 'Probability'}, inplace=True)


##
import pickle

with open('DFPost_POST_RISK_Hetero.pkl', 'wb') as f:
    pickle.dump(DFPost_POST_RISK_Hetero, f)
##



import seaborn as sns

sns.violinplot(DFPost_POST_RISK_Hetero, x="HeteroNorm", y="Probability")
plt.xlabel("HeteroNorm")
plt.title('Probability of HighRiskProfile')
plt.ylabel("Probability")

plt.show()




### GENDER / RISK / HIV


dat_list = {
    'HIV': PKD_model_DF.HIV.values, # 2
    'HighRiskHIVNoGen': PKD_model_DF.HighRiskHIVNoGen.values, # 2
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values, # 2
}



def model(HIV, Płeć_encoded, HighRiskHIVNoGen=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))

    logit_p = a  + b[Płeć_encoded] + c[HighRiskHIVNoGen]

    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)
                                                                
HIv_risk_gender = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIv_risk_gender.run(random.PRNGKey(0), **dat_list)
HIv_risk_gender.print_summary(0.89)


POST_HIv_risk_gender = HIv_risk_gender.get_samples()

DFPOST_HIv_risk_gender = pd.DataFrame()


for v in range(2): # Loop for b
    postDFOne = vmap(lambda k: expit(POST_HIv_risk_gender["a"] + POST_HIv_risk_gender["b"][:, v] +
                                      POST_HIv_risk_gender["c"][:, k]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=[0, 1]) 
    postDFLong = pd.melt(postDFOne)
    postDFLong['Gender'] = jnp.repeat(v, len(postDFLong))
    DFPOST_HIv_risk_gender = pd.concat([DFPOST_HIv_risk_gender, postDFLong])


DFPOST_HIv_risk_gender.head()

DFPOST_HIv_risk_gender.rename(columns={'variable': 'RiskFactor', 'value': 'Probability'}, inplace=True)


##
import pickle

with open('DFPOST_HIv_risk_gender.pkl', 'wb') as f:
    pickle.dump(DFPOST_HIv_risk_gender, f)
    
##

with open('DFPOST_HIv_risk_gender.pkl', 'rb') as f:
    DFPOST_HIv_risk_gender = pickle.load(f)


import seaborn as sns

g = sns.FacetGrid(DFPOST_HIv_risk_gender, col='Gender')
g.map_dataframe(sns.violinplot, x="RiskFactor", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()
