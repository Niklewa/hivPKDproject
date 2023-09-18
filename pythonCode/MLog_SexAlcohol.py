import ModelBuildingPackages


PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_model_DF.csv")

PKD_model_DF.columns

pd.set_option('display.max_columns', None)
PKD_model_DF.head


#################### Data preparation

PKD_model_DF['Seks_alkohol'].unique()
PKD_model_DF["Seks_alkohol"].isna().sum()

PKD_model_DF.dropna(subset=['Seks_alkohol'], inplace=True)

PKD_model_DF['Hetero_normative'].unique()



dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'Seks_alkohol': PKD_model_DF.Seks_alkohol.values.astype('int64'), # 2
    'Hetero_normative': PKD_model_DF.Hetero_normative.values, # 2
}


# Model

def model( Seks_alkohol, Hetero_normative, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))

    logit_p = a + b[Seks_alkohol] + c[Hetero_normative] 
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_AlcoholHetero = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_AlcoholHetero.run(random.PRNGKey(0), **dat_list)
HIV_AlcoholHetero.print_summary(0.89)



Post_HIVAlocholHetero = HIV_AlcoholHetero.get_samples()


postDF_AlcHetero = pd.DataFrame()

for n in range(2):  
    postDFOne = vmap(lambda k: expit(Post_HIVAlocholHetero["a"] + Post_HIVAlocholHetero["c"][:, k] + 
                                         Post_HIVAlocholHetero["b"][:, n]), 0, 1)(jnp.arange(2))
    postDFOne = pd.DataFrame(postDFOne, columns=['0', '1']) 
    postDFLong = pd.melt(postDFOne)
    postDFLong['SexAlcohol'] = jnp.repeat(n, len(postDFLong))
    postDF_AlcHetero = pd.concat([postDF_AlcHetero, postDFLong])

postDF_AlcHetero.rename(columns={'variable': 'HeteroNorm', 'value': 'Probability'}, inplace=True)
postDF_AlcHetero.head()

# VIS

import seaborn as sns

sns.set(style="whitegrid")

g = sns.FacetGrid(postDF_AlcHetero, col='SexAlcohol')
g.map_dataframe(sns.violinplot, x="HeteroNorm", y="Probability")
plt.xlabel("")
plt.ylabel("Probability")

plt.show()

###############################

# reload the data !


PKD_model_DF["PrEP"].isna().sum()
PKD_model_DF.dropna(subset=['PrEP'], inplace=True)

PKD_model_DF['Hetero_normative'].unique()


PKD_model_DF = PKD_model_DF[PKD_model_DF['Płeć'] != 'I']
label_encoder = LabelEncoder()
PKD_model_DF['Płeć_encoded'] = label_encoder.fit_transform(PKD_model_DF['Płeć'])
# M 2 / K 1 / I 0 
PKD_model_DF['Płeć_encoded'].unique()


dat_list = {
    'HIV': PKD_model_DF.HIV.values,  # 2
    'PrEP': PKD_model_DF.PrEP.values.astype('int64'), # 2
    'Hetero_normative': PKD_model_DF.Hetero_normative.values, # 2
    'Płeć_encoded': PKD_model_DF.Płeć_encoded.values,
    
}


# Model

def model(Płeć_encoded, PrEP, Hetero_normative, HIV=None):     
    a = numpyro.sample("a", dist.Normal(0, 1.5))  
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([2]))
    c = numpyro.sample("c", dist.Normal(0, 0.5).expand([2]))
    d = numpyro.sample("d", dist.Normal(0, 0.5).expand([2]))

    logit_p = a + b[PrEP] + c[Hetero_normative] + d[Płeć_encoded]
    numpyro.sample("HIV", dist.Binomial(logits=logit_p), obs=HIV)

                                                                
HIV_PrEPHeteroGender = MCMC(NUTS(model), num_warmup=500, num_samples=500) 
HIV_PrEPHeteroGender.run(random.PRNGKey(0), **dat_list)
HIV_PrEPHeteroGender.print_summary(0.89)



Post_HIVPrepSexlHetero = HIV_PrEPHeteroGender.get_samples()


PostDF_HIVPrepSexlHetero = pd.DataFrame()

for n in range(2):  # Loop for "d" categories
    for m in range(2):  # Loop for "b" categories
        postDFOne = vmap(lambda k: expit(Post_HIVPrepSexlHetero["a"] + Post_HIVPrepSexlHetero["c"][:, k] + 
                                         Post_HIVPrepSexlHetero["d"][:, n] + Post_HIVPrepSexlHetero["b"][:, m]), 0, 1)(jnp.arange(2))
        postDFOne = pd.DataFrame(postDFOne, columns=[0, 1]) # 
        postDFLong = pd.melt(postDFOne)
        postDFLong['Gender'] = jnp.repeat(n, len(postDFLong))
        postDFLong['PrEP'] = jnp.repeat(m, len(postDFLong))
        PostDF_HIVPrepSexlHetero = pd.concat([PostDF_HIVPrepSexlHetero, postDFLong])



sanityCheck = PostDF_HIVPrepSexlHetero.groupby(['variable', 'PrEP', 'Gender']).size().reset_index(name='Count')

PostDF_HIVPrepSexlHetero.rename(columns={'variable': 'HeteroNorm', 'value': 'Probability'}, inplace=True)


### Vis


import seaborn as sns
from matplotlib.lines import Line2D

sns.set(style="whitegrid")

custom_palette = ["#8D4585", "#56B4E9"] 

postDF_Gender1 = PostDF_HIVPrepSexlHetero[PostDF_HIVPrepSexlHetero['Gender'] == 1]

postDF_Gender0 = PostDF_HIVPrepSexlHetero[PostDF_HIVPrepSexlHetero['Gender'] == 0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))  
# Plot for gender = 1
sns.violinplot(data=postDF_Gender1, x="PrEP", y="Probability", hue="HeteroNorm", ax=axes[0], palette=custom_palette)
axes[0].set_title("Males")
axes[0].set_xlabel("")
axes[0].set_ylabel("Probability")

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[0], markersize=10, label='Non hetero'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[1], markersize=10, label='Hetero')]


legend2 = axes[0].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# Plot for gender = 0
sns.violinplot(data=postDF_Gender0, x="PrEP", y="Probability", hue="HeteroNorm", ax=axes[1], palette=custom_palette)
axes[1].set_title("Females")
axes[1].set_xlabel("On PrEP")
axes[1].set_ylabel("Probability")


lelegend2 = axes[1].legend(title="", loc="upper right", handles=legend_handles)
legend2.get_title().set_fontsize('10')
legend2.get_frame().set_alpha(None)

# unifying y axis values
combined_data = PostDF_HIVPrepSexlHetero["Probability"]
y_min = combined_data.min()
y_max = combined_data.max()
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

plt.suptitle("Probability of Getting HIV Infected")

plt.tight_layout()
plt.show()
