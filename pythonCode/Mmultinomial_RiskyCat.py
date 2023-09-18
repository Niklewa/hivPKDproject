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

from sklearn.preprocessing import LabelEncoder

PKD_model_DF= pd.read_csv('PKD_model_DF.csv')

pd.set_option('display.max_columns', None)
PKD_model_DF.head

DF_multinomial = PKD_model_DF[PKD_model_DF['ShortPartnersAmount'] != 'na']
DF_multinomial['ShortPartnersAmount'].unique()

DF_multinomial = PKD_model_DF[PKD_model_DF['Płeć'] != 'I']
label_encoder = LabelEncoder()
DF_multinomial['Płeć_encoded'] = label_encoder.fit_transform(DF_multinomial['Płeć'])
# M 1 / K 0
DF_multinomial['Płeć_encoded'].unique()

encoded_target = pd.get_dummies(DF_multinomial['ShortPartnersAmount'], prefix='category')
encoded_target = encoded_target.astype(int)

X = DF_multinomial[['Płeć_encoded', 'Hetero_normative']]

X = pd.concat([X, encoded_target], axis=1)




def multinomial_model(X, y):
    num_features = X.shape[1]
    num_categories = y.shape[1]  # Number of categories

    beta = numpyro.sample('beta', dist.Normal(0, 1).expand([num_features, num_categories]))
    logits = jnp.dot(X, beta)

    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Categorical(logits=logits), obs=y)



def multinomial_model(X, y):
    num_features = X.shape[1]
    num_categories = y.shape[1]

    beta = numpyro.sample('beta', dist.Normal(0, 1).expand([num_features, num_categories]))
    logits = jnp.dot(X, beta)

    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Categorical(logits=logits), obs=y)

one_hot_encoded_target = encoded_target[['category_1-10', 'category_11-50', 'category_above_51']].values
X_features = X[['Płeć_encoded', 'Hetero_normative']].values



Multi_model = MCMC(NUTS(multinomial_model), num_warmup=500, num_samples=500) 
Multi_model.run(random.PRNGKey(0), X_features, **one_hot_encoded_target)
Multi_model.print_summary(0.89)


#### my version

X_list = {
    'Płeć_encoded': X.Płeć_encoded.values, # 2
    'Hetero_normative': X.Hetero_normative.values, # 2
}

X_list_jax = {
    'Hetero_normative': jnp.array(X_list['Hetero_normative'])
}

one_hot_encoded_target = encoded_target[['category_1-10', 'category_11-50', 'category_above_51']].values
one_hot_encoded_target_jax = jnp.array(one_hot_encoded_target)

X_features = X[[ 'Hetero_normative']].values

from jax import jit

@jit

def model( Hetero_normative, Target=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b_hetero = numpyro.sample("b_hetero", dist.Normal(0, 0.5).expand([2]))
    
    logits =  b_hetero[Hetero_normative]
    probs = nn.softmax(logits, axis=-1)
    
    numpyro.sample("category", dist.Categorical(probs=probs), obs=Target)


Multi_model = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
Multi_model.run(random.PRNGKey(0), **X_list_jax, Target=one_hot_encoded_target_jax)
Multi_model.print_summary()

Multi_model.get_samples()
