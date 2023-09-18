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

if "SVG" in os.environ:
    %config InlineBackend.figure_formats = ["svg"]
warnings.formatwarning = lambda message, category, *args, **kwargs: "{}: {}\n".format(
    category.__name__, message
)

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


##### data base import ####################################################

PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_model_DF.csv")

PKD_model_DF.columns

pd.set_option('display.max_columns', None)
PKD_model_DF.head


###########

PKD_model_DF['LongPartnersAmount'].unique()
PKD_model_DF['ShortPartnersAmount'].unique()


df_model = PKD_model_DF.dropna(subset=['LongPartnersAmount'])

category_mapping = {
    '1': 0,
    '2-5': 1,
    '6-10': 2,
    '11-20': 3,
    '21-50': 4,
    '51-100': 5,
    '>101': 6
}

df_model.loc[:, 'LongPartnersAmount'] = df_model['LongPartnersAmount'].replace(category_mapping)

df_model['LongPartnersAmount'].unique()


from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
df_model['LongPartnersAmount_enc'] = label_encoder.fit_transform(df_model['LongPartnersAmount'])
df_model['LongPartnersAmount_enc'].unique()


















dat_list = {
    'HIV': df_model.HIV.values,
    'LongPartnersAmount_enc': df_model.LongPartnersAmount_enc.values,
    }


def model(LongPartnersAmount_enc, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([7]))
    logit_p = a + b[LongPartnersAmount_enc]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_Partners = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_Partners.run(random.PRNGKey(0), **dat_list)
HIV_Partners.print_summary(0.89)



postPartners_ = HIV_Partners.get_samples()

##
import pickle

with open('post.pkl', 'wb') as f:
    pickle.dump(postPartners, f)
##

postPartners["log_lik"] = log_likelihood(HIV_Partners.sampler.model, postPartners, **dat_list)["HIV"]
{k: v.shape for k, v in postPartners.items()}


# Post analysis

postProbs = vmap(lambda k: expit(postPartners["a"] + postPartners["b"][:, k]), 0, 1)(jnp.arange(7))

postDF = pd.DataFrame(postProbs, columns=['1', '2-5', '6-10', '11-20',  '21-50', '51-100', '>101'])

postDFLong = pd.melt(postDF, var_name="PartnersNum", value_name="Probability")


# visualization
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.violinplot(x="PartnersNum", y="Probability", data=postDFLong)
sns.stripplot(x="PartnersNum", y="Probability", data=postDFLong,
               jitter=True, color="lightblue", alpha=0.1)

plt.xlabel("Number of Partners")
plt.ylabel("Probability")
plt.title("Probability of Getting Infected")

plt.show()

# model Longpartners and heteronorm




dat_list = {
    'HIV': df_model.HIV.values,
    'LongPartnersAmount_enc': df_model.LongPartnersAmount_enc.values,
    'Hetero_normative': df_model.Hetero_normative.values,
    }


def model(LongPartnersAmount_enc, Hetero_normative, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([7]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))
    logit_p = a + b[LongPartnersAmount_enc] + c[Hetero_normative]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_Partners_Hetero = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_Partners_Hetero.run(random.PRNGKey(0), **dat_list)
HIV_Partners_Hetero.print_summary(0.89)



POSTHIV_Partners_Hetero = HIV_Partners_Hetero.get_samples()



##
import pickle

with open('POSTHIV_Partners_Hetero.pkl', 'wb') as f:
    pickle.dump(POSTHIV_Partners_Hetero, f)
##

postDF_Hetero_Partners = pd.DataFrame()

for n in range(7):
    postDFOne = vmap(lambda k: expit(POSTHIV_Partners_Hetero["a"] + POSTHIV_Partners_Hetero["b"][:, n] + POSTHIV_Partners_Hetero["c"][:, k]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=['Non_hetero', 'Hetero'])
    postDFLong = pd.melt(postDFOne)
    postDFLong['Partners'] = jnp.repeat(n, len(postDFLong)) 
    postDF_Hetero_Partners = pd.concat([postDF_Hetero_Partners, postDFLong])


#category_mapping = {
#'1': 0,
#'2-5': 1,
#'6-10': 2,
#'11-20': 3,
#'21-50': 4,
#'51-100': 5,
#'>101': 6
#}



postDF_Hetero_Partners.rename(columns={'variable': 'Hetero_norm', 'value': 'Probability'}, inplace=True)
postDF_Hetero_Partners.head()



# visualization

import seaborn as sns

g = sns.FacetGrid(postDF_Hetero_Partners, col='Hetero_norm')
g.map_dataframe(sns.violinplot, x="Partners", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()










#### Trying multinomial logistic regression (softmax) # NOT WORKING


df_model['LongPartnersAmount'].unique()


dat_listSoftmax = {
    'Hetero_normative': df_model.Hetero_normative.values,
    'LongPartnersAmount': df_model.LongPartnersAmount.values,
    }


def model(LongPartnersAmount, Hetero_normative):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    logit_p = a + b[Hetero_normative]
    numpyro.sample("LongPartnersAmount", dist.Multinomial(logits=logit_p), obs=LongPartnersAmount)

                                                                
HIV_Partners = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_Partners.run(random.PRNGKey(0), **dat_listSoftmax)
HIV_Partners.print_summary(0.89)

postPartnersSoftmax = HIV_Partners.get_samples()