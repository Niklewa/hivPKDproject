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


##### data base import ####################################################

PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_model_DF.csv")

PKD_model_DF.columns

pd.set_option('display.max_columns', None)
PKD_model_DF.head

# let's first build simple logistic regression model with small amount of predictors


# HIV 0,1 / Hetero_normative 0,1 (how sexual identity influence being hiv positive)

PKD_model_DF["HIV"].unique()
PKD_model_DF["Hetero_normative"].unique()

len(PKD_model_DF)

dat_list = {
    'HIV': PKD_model_DF.HIV.values,
    'Hetero_normative': PKD_model_DF.Hetero_normative.values,
    }


def model(Hetero_normative, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  # well the prob of being HIV infected is
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2])) # very small, skew it?
    logit_p = a + b[Hetero_normative]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_HetreoNorm = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_HetreoNorm.run(random.PRNGKey(0), **dat_list)
HIV_HetreoNorm.print_summary(0.89)


# Let's check priors first:

# priors difference
prior = Predictive(model, num_samples=int(1e4))(
    random.PRNGKey(1999), HIV=0, Hetero_normative=0
)
priorsINV = vmap(lambda k: expit(prior["a"] + prior["b"][:, k]), 0, 1)(jnp.arange(2))
jnp.mean(jnp.abs(priorsINV[:, 0] - priorsINV[:, 1]))


# priors dist

az.plot_kde(jnp.abs(priorsINV[:, 0] - priorsINV[:, 1]), bw=0.3)
plt.show() # not sure what does it mean

# a 
az.plot_kde(expit(prior["a"]))
plt.show()

# b
az.plot_kde(expit(prior["b"]))
plt.show()


# Posterior analysis

post_HIV_HetreoNorm = HIV_HetreoNorm.get_samples()

## let's save it
import pickle

with open('post_HIV_HetreoNorm.pkl', 'wb') as f:
    pickle.dump(post_HIV_HetreoNorm, f)

##
#with open('post.pkl', 'rb') as f:
#    post = pickle.load(f)


# Showing structure of the data
post_HIV_HetreoNorm["log_lik"] = log_likelihood(HIV_HetreoNorm.sampler.model, post_HIV_HetreoNorm, **dat_list)["HIV"]
{k: v.shape for k, v in post_HIV_HetreoNorm.items()}


# To Probabilities
post_HIV_HetreoNorm

postDFOne = vmap(lambda k: expit(post_HIV_HetreoNorm["a"] + post_HIV_HetreoNorm["b"][:, k]), 0, 1)(jnp.arange(2))
postDFOne = pd.DataFrame(postDFOne, columns=['non hetero', 'hetero']) 
DFPost_post_HIV_HetreoNorm = pd.melt(postDFOne)

DFPost_post_HIV_HetreoNorm.head()
DFPost_post_HIV_HetreoNorm.rename(columns={'variable': 'HeteroNorm', 'value': 'Probability'}, inplace=True)

### Visualization of results

import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.violinplot(x="HeteroNorm", y="Probability", data=DFPost_post_HIV_HetreoNorm)

sns.stripplot(x="HeteroNorm", y="Probability", data=DFPost_post_HIV_HetreoNorm,
               jitter=True, color="black", alpha=0.1)


plt.ylim(0, 0.10)
plt.xlabel("Hetero Normative")
plt.ylabel("Probability")
plt.title("Probability of Getting Infected")




plt.show()



# Loading the backup data file 

pickle_file_path = "pythonCode/post.pkl"

import pickle

with open(pickle_file_path, 'rb') as file:
   post = pickle.load(file)



# Model with gender added

PKD_model_DF.head()
PKD_model_DF['Płeć'].unique()

PKD_model_DF = PKD_model_DF[PKD_model_DF["Płeć"] != 'I']


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
PKD_model_DF['Płeć_encoded'] = label_encoder.fit_transform(PKD_model_DF['Płeć'])
# 1, 0  M, K 
PKD_model_DF['Płeć_encoded'].unique()


dat_listGender = {
    'HIV': PKD_model_DF.HIV.values,
    'Hetero_normative': PKD_model_DF.Hetero_normative.values,
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values,
    }


def model_with_Gender(Hetero_normative, Płeć_encoded, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2])) 
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([3])) 

    logit_p = a + b[Hetero_normative] + c[Płeć_encoded]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_HetreoNorm_Gender = MCMC(NUTS(model_with_Gender), num_warmup=500, num_samples=500) 
HIV_HetreoNorm_Gender.run(random.PRNGKey(0), **dat_listGender)
HIV_HetreoNorm_Gender.print_summary(0.89)





# post analysis

post_Hetero_Gender = HIV_HetreoNorm_Gender.get_samples()

##
import pickle

with open('post_Hetero_Gender.pkl', 'wb') as f:
    pickle.dump(post_Hetero_Gender, f)
##


with open('post_Hetero_Gender.pkl', 'rb') as f:
    post_Hetero_Gender = pickle.load(f)



postDF_Hetero_Gender = pd.DataFrame()
for n in range(2):
    postDFOne = vmap(lambda k: expit(post_Hetero_Gender["a"] + post_Hetero_Gender["b"][:, k] +
                                     post_Hetero_Gender["c"][:, n]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=['Non-hetero', 'Hetero'])
    postDFLong = pd.melt(postDFOne)
    postDFLong['Gender'] = jnp.repeat(n, len(postDFLong)) # 2, 1, 0 ???
    postDF_Hetero_Gender = pd.concat([postDF_Hetero_Gender, postDFLong])



postDF_Hetero_Gender = postDF_Hetero_Gender.rename(columns={'variable': 'SexIdent', 'value': 'Probability'})

import pickle

with open('postDF_Hetero_Gender.pkl', 'wb') as f:
    pickle.dump(postDF_Hetero_Gender, f)

# visualization

import seaborn as sns


sns.set(style="whitegrid")
g = sns.FacetGrid(postDF_Hetero_Gender, col='Gender')
g.map_dataframe(sns.violinplot, x="Hetero_norm", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()


################################



