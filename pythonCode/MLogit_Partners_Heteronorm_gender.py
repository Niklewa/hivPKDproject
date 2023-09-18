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

PKD_model_DF['ShortPartnersAmount'].unique()
PKD_model_DF['Hetero_normative'].unique()
PKD_model_DF['Płeć'].unique()


len(PKD_model_DF)

df_model = PKD_model_DF.dropna(subset=['ShortPartnersAmount'])
df_model = df_model[df_model['ShortPartnersAmount'] != 'na']

df_model = df_model[df_model['Płeć'] != 'I']

len(df_model)

df_model['ShortPartnersAmount'].unique()



df_model['ShortPartnersAmount'].unique()
df_model['Płeć'].unique()

df_model['HIV'].unique()
# model building


# using encoder instead

from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
df_model['Płeć_encoded'] = label_encoder.fit_transform(df_model['Płeć'])
# M 1 / K 0

df_model['Płeć_encoded'].unique()



df_model['ShortPartnersAmount_enc'] = label_encoder.fit_transform(df_model['ShortPartnersAmount'])
# '1-10' /  0, '11-50'  / 1, 'above_51' / 2

df_model['ShortPartnersAmount_enc'].unique()




dat_list = {
    'HIV': df_model.HIV.values,
    'ShortPartnersAmount_enc': df_model.ShortPartnersAmount_enc.values,  
    'Hetero_normative': df_model.Hetero_normative.values,
    'Płeć_encoded': df_model.Płeć_encoded.values,
}



def model( Płeć_encoded, ShortPartnersAmount_enc, Hetero_normative, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([3]))
    d = numpyro.sample("d", dist.Normal(0, 0.5).expand([2]))
    logit_p = a + b[Płeć_encoded] + c[ShortPartnersAmount_enc] + d[Hetero_normative]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_3PartGenHetero = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_3PartGenHetero.run(random.PRNGKey(0), **dat_list)
HIV_3PartGenHetero.print_summary(0.89)



Part3GenHetero = HIV_3PartGenHetero.get_samples()

##
import pickle

with open('Part3GenHeteroPost.pkl', 'wb') as f:
    pickle.dump(Part3GenHetero, f)
##



postDF_Part3GenHetero = pd.DataFrame()



for n in range(2):  # Loop for "d" categories
    for m in range(2):  # Loop for "b" categories
        postDFOne = vmap(lambda k: expit(Part3GenHetero["a"] + Part3GenHetero["c"][:, k] + Part3GenHetero["d"][:, n] + Part3GenHetero["b"][:, m]), 0, 1)(jnp.arange(3))
        postDFOne = pd.DataFrame(postDFOne, columns=['1-10',  '11-50', 'above_51'])
        postDFLong = pd.melt(postDFOne)
        postDFLong['HeteroNorm'] = jnp.repeat(n, len(postDFLong))
        postDFLong['Gender'] = jnp.repeat(m, len(postDFLong))
        postDF_Part3GenHetero = pd.concat([postDF_Part3GenHetero, postDFLong])

# sanity check
combination_counts = postDF_Part3GenHetero.groupby(['variable', 'HeteroNorm', 'Gender']).size().reset_index(name='Count')



# visualization

import seaborn as sns
from matplotlib.lines import Line2D

postDF_Part3GenHetero.head()

postDF_Part3GenHetero.rename(columns={'variable': 'PartnersNum', 'value': 'Probability'}, inplace=True)


sns.set(style="whitegrid")
g = sns.FacetGrid(postDF_Part3GenHetero, col='PartnersNum')
g.map_dataframe(sns.violinplot, x="HeteroNorm", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()



sns.set(style="whitegrid")

custom_palette = ["#009E73", "#56B4E9"] 

postDF_Gender1 = postDF_Part3GenHetero[postDF_Part3GenHetero['Gender'] == 1]

postDF_Gender0 = postDF_Part3GenHetero[postDF_Part3GenHetero['Gender'] == 0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))  
# Plot for gender = 1
sns.violinplot(data=postDF_Gender1, x="PartnersNum", y="Probability", hue="HeteroNorm", ax=axes[0], palette=custom_palette)
axes[0].set_title("Males")
axes[0].set_xlabel("")
axes[0].set_ylabel("Probability")

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[0], markersize=10, label='Non hetero'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[1], markersize=10, label='Hetero')]


legend2 = axes[0].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# Plot for gender = 0
sns.violinplot(data=postDF_Gender0, x="PartnersNum", y="Probability", hue="HeteroNorm", ax=axes[1], palette=custom_palette)
axes[1].set_title("Females")
axes[1].set_xlabel("Amount of Partners")
axes[1].set_ylabel("Probability")


lelegend2 = axes[1].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# unifying y axis values
combined_data = postDF_Part3GenHetero["Probability"]
y_min = combined_data.min()
y_max = combined_data.max()
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

plt.suptitle("Probability of Getting HIV Infected")

plt.tight_layout()
plt.show()







