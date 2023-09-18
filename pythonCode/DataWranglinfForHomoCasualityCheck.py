from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


import jax.numpy as jnp
from jax import random, vmap

#############



HIVdata = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKDjoint.csv")

HIVdata.columns

len(HIVdata)

pd.set_option('display.max_columns', None)
HIVdata.head

# out of the whole DF let's take only columns signifcant to us


# Płeć, Powód, TPR_wynik TPO_wynik, Data_przeprowadzenia_rozmowy
#Orientacja, Partner_inny_rok, Kontakty_seks_rok, Kontakty_seks_rok, Kontakty_seks_rok_anal_active_condom, Kontakty_seks_rok_anal_condom
# Kontakty_seks_rok_anal_passive, Kontakty_seks_rok_condom,  Kontakty_seks_rok_inne,  Kontakty_seks_rok_inne_condom
# Kontakty_seks_rok_oral_active, Kontakty_seks_rok_oral_active_condom, Kontakty_seks_rok_oral_passive, Kontakty_seks_rok_oral_passive_condom,
# Partner_inny_rok,Partner_inny_rok,  Partner_inny_rok_liczba, Partner_rok_K, Partner_rok_M, Partner_stały_rok_liczba, Partner_życie_liczba, Partner_życie_wielu
# Powód_k._IDU


HIVdata

PKD_small = HIVdata[[

"Płeć", "Powód", "TPR_wynik", "TPO_wynik", "Data_przeprowadzenia_rozmowy",
"Orientacja", "Partner_inny_rok", "Kontakty_seks_rok", 
 "Kontakty_seks_rok_anal_active_condom", "Kontakty_seks_rok_anal_condom",
"Kontakty_seks_rok_anal_passive", "Kontakty_seks_rok_condom",
 "Kontakty_seks_rok_inne", "Kontakty_seks_rok_inne_condom",
"Kontakty_seks_rok_oral_active", "Kontakty_seks_rok_oral_active_condom",
 "Kontakty_seks_rok_oral_passive", "Kontakty_seks_rok_oral_passive_condom",
"Partner_inny_rok", "Partner_inny_rok", "Partner_inny_rok_liczba",
 "Partner_rok_K", "Partner_rok_M", "Partner_stały_rok_liczba", 
 "Partner_życie_liczba", "Partner_życie_wielu",
"Powód_k._IDU", 'Kontakty_seks_rok_wagin.', "Iniekcje", 'PrEP', 'Seks_alkohol'

]]

len(PKD_small)


# adding Year column

PKD_small[:, 'Year'] = PKD_small.loc[ 'Data_przeprowadzenia_rozmowy'].dt.year

PKD_small.drop(columns=['Data_przeprowadzenia_rozmowy'], inplace=True)

# adding column HIV infected, basing on TPR_wynik and TPO_wynik 

# IDU - injective drug user
# STP - quick test
# TPR - normal test
# TPO - confirmation test (after the first one being positive)

PKD_small["TPR_wynik"].unique()
PKD_small["TPO_wynik"].unique()

PKD_small.dropna(subset=['TPR_wynik'], inplace=True)

PKD_small.loc[:, 'TPR_wynik'] = PKD_small['TPR_wynik'].replace({'u': 0, 'd': 1})
PKD_small.loc[:, 'TPO_wynik'] = PKD_small['TPO_wynik'].replace({'u': 0, 'd': 1})

PKD_small.loc[:, 'TPO_wynik'].fillna(0, inplace=True)


# values   w (weak 148);  i (indeterminate 35) may cause a problem,
# let's just delete those persons from the data set, to have only confirm positive/negative cases

PKD_small = PKD_small[~PKD_small['TPO_wynik'].isin(['i', 'w'])]

PKD_small["TPO_wynik"].unique()

len(PKD_small)

# adding HIV positive column

PKD_small.loc[:, 'HIV'] = (PKD_small['TPO_wynik'] == 1).astype(int)


count_tpo = (PKD_small['TPO_wynik'] == 1).sum() 
count_tpr = (PKD_small['TPR_wynik'] == 1).sum()

count_tpo / count_tpr # 85.3% specificity ############################# 14.7% od positive tests were false


# Creating Hetero_normative value

(PKD_small["Orientacja"]).unique()

# let's delete: 'wybierz', nan, 'odmowa odpowiedzi'

PKD_small.dropna(subset=['Orientacja'], inplace=True)

PKD_small = PKD_small[~PKD_small['Orientacja'].isin(['wybierz', 'odmowa odpowiedzi'])]

PKD_small.loc[:, 'Hetero_normative'] = np.where(PKD_small['Orientacja'] == 'heteroseksualna', 1, 0)

(PKD_small['Hetero_normative'] == 0).sum() /  (PKD_small['Hetero_normative'] == 1).sum() # 0.43 are non hetero


# Creating Anal sex variable, and condom use

(PKD_small["Kontakty_seks_rok_anal_active_condom"]).unique()
(PKD_small["Kontakty_seks_rok_anal_condom"]).unique()
(PKD_small["Kontakty_seks_rok_anal_passive"]).unique()

# Anal - pass, active, vers, no

conditionsAnal = [
    (PKD_small['Kontakty_seks_rok_anal_passive'] == 1) & (PKD_small['Kontakty_seks_rok_anal_active_condom'].isin(['zawsze', 'czasami', 'nigdy'])),
    (PKD_small['Kontakty_seks_rok_anal_passive'] == 1),
    (PKD_small['Kontakty_seks_rok_anal_active_condom'].isin(['zawsze', 'czasami', 'nigdy']))
]
choicesAnal = ['vers', 'passive', 'active']

PKD_small.loc[:, 'Anal'] = np.select(conditionsAnal, choicesAnal, default='no')

(PKD_small["Anal"]).unique()

# AnalProtec - always, sometimes, never, noAnal

conditionsAnalProtec = [
    (PKD_small['Kontakty_seks_rok_anal_condom'] == 'zawsze') | (PKD_small['Kontakty_seks_rok_anal_active_condom'] == 'zawsze'),
    (PKD_small['Kontakty_seks_rok_anal_condom'] == 'czasami') | (PKD_small['Kontakty_seks_rok_anal_active_condom'] == 'czasami'),
    (PKD_small['Kontakty_seks_rok_anal_condom'] == 'nigdy') | (PKD_small['Kontakty_seks_rok_anal_active_condom'] == 'nigdy')
]
choicesAnalProtec = ['always', 'sometimes', 'never']

PKD_small.loc[:, 'AnalProtec'] = np.select(conditionsAnalProtec, choicesAnalProtec, default='noAnal')

(PKD_small["AnalProtec"]).unique()

# General condom use

#Kontakty_seks_rok_condom
#Kontakty_seks_rok_wagin.


# ManyPartners

#Partner_życie_liczba
#Partner_stały_rok_liczba
#Partner_inny_rok_liczba

(PKD_small["Partner_życie_liczba"]).unique()
(PKD_small["Partner_stały_rok_liczba"]).unique()
(PKD_small["Partner_inny_rok_liczba"]).unique()

# ShortPartnersAmount '1-10', '11-50', 'above_51'

conditionsPartners = [
    (PKD_small['Partner_życie_liczba'].isin(['1', '2-5', '6-10'])),
    (PKD_small['Partner_życie_liczba'].isin(['11-20', '21-50'])),
    (PKD_small['Partner_życie_liczba'].isin(['51-100', '>101'])),
]
choicesPartners = ['1-10', '11-50', 'above_51']

PKD_small.loc[:, 'ShortPartnersAmount'] = np.select(conditionsPartners, choicesPartners, default='na')

(PKD_small["ShortPartnersAmount"]).unique()


# For oral sex
PKD_small.head()

(PKD_small["Kontakty_seks_rok_oral_passive_condom"]).unique()
(PKD_small["Kontakty_seks_rok_oral_active_condom"]).unique()
(PKD_small["Kontakty_seks_rok_oral_active"]).unique()
(PKD_small["Kontakty_seks_rok_oral_passive"]).unique()

# Oral - pass, active, vers, no

conditionsOral = [
    (PKD_small['Kontakty_seks_rok_oral_passive'] == 1) & (PKD_small['Kontakty_seks_rok_oral_active'] == 1),
    (PKD_small['Kontakty_seks_rok_anal_passive'] == 1),
    (PKD_small['Kontakty_seks_rok_oral_active'] == 1)
]
choicesOral = ['vers', 'passive', 'active']

PKD_small.loc[:, 'Oral'] = np.select(conditionsOral, choicesOral, default='no')

(PKD_small["Oral"]).unique()

# OralProtec - always, sometimes, never, noAnal

conditionsoralProtec = [
    (PKD_small['Kontakty_seks_rok_oral_active_condom'] == 'zawsze') | (PKD_small['Kontakty_seks_rok_oral_passive'] == 'zawsze'),
    (PKD_small['Kontakty_seks_rok_oral_active_condom'] == 'czasami') | (PKD_small['Kontakty_seks_rok_oral_passive'] == 'czasami'),
    (PKD_small['Kontakty_seks_rok_oral_active_condom'] == 'nigdy') | (PKD_small['Kontakty_seks_rok_oral_passive'] == 'nigdy')
]
choicesOralProtec = ['always', 'sometimes', 'never']

PKD_small.loc[:, 'OralProtec'] = np.select(conditionsAnalProtec, choicesOralProtec, default='noOral')

(PKD_small["OralProtec"]).unique()






# LongPartnersAmount

PKD_small['Partner_życie_liczba'].replace(['nie wiem', 'odmowa odpowiedzi'], float('nan'), inplace=True)

PKD_small['LongPartnersAmount'] = PKD_small['Partner_życie_liczba']


# Saving wrangled data frame
PKD_small.to_csv('PKD_model_DF.csv', index=False)


############## 06.08


HIVdata = pd.read_csv('PKD_model_DF.csv')

HIVdata.columns

pd.set_option('display.max_columns', None)
HIVdata.head



# vaginalProt = yesAlways, yesSometimes, yesNever, no - # Kontakty_seks_rok_wagin. and Kontakty_seks_rok_condom

HIVdata['Kontakty_seks_rok_condom'].unique()
HIVdata['Kontakty_seks_rok_wagin.'].unique() 


HIVdata['Kontakty_seks_rok_condom'].isna().sum()

HIVVaginProtk = HIVdata.groupby(['Kontakty_seks_rok_condom']).size().reset_index(name='Count')

HIVdata.loc[:, 'VagSexProt'] = HIVdata['Kontakty_seks_rok_condom'].replace({'czasami': 'sometimes', 'zawsze': 'always',
                                                   'nigdy': 'never', 'nie dotyczy': 'NaN', 'odmowa odpowiedzi': 'NaN'})

HIVdata['VagSexProt'] = HIVdata['VagSexProt'].fillna('noVagSex')


# PrEP

HIVdata['PrEP'].unique()

HIVPREPCheck = HIVdata.groupby(['PrEP']).size().reset_index(name='Count')
HIVdata.loc[:, 'PrEP'] = HIVdata['PrEP'].replace({'NIE': 0, 'TAK': 1})
HIVdata['PrEP'].replace(['odmowa odpowiedzi'], float('NaN'), inplace=True)
HIVdata['PrEP'].unique()

#Seks_alkohol

HIVAlc_Check = HIVdata.groupby(['Seks_alkohol']).size().reset_index(name='Count')
HIVdata['Seks_alkohol'].replace(['nie pamiętam', 'odmowa odpowiedzi', 'wybierz'], float('NaN'), inplace=True)
HIVdata.loc[:, 'Seks_alkohol'] = HIVdata['Seks_alkohol'].replace({'NIE': 0, 'TAK': 1})

# Iniekcje - IDU (injective drug user)

HIVdata.loc[:, 'IDU'] = HIVdata['Iniekcje'].replace({'NIE': 0, 'TAK': 1})

HIVdata['IDU'].replace(['nie dotyczy', 'odmowa odpowiedzi', 'wybierz'], float('NaN'), inplace=True)

HIVIDUCheck = HIVdata.groupby(['IDU']).size().reset_index(name='Count')
HIVIDUCheck

HIVdata.to_csv('PKD_model_DF.csv', index=False)



### Risk profile variable 09.08

HIVdata = pd.read_csv('PKD_model_DF.csv')

HIVdata.columns

pd.set_option('display.max_columns', None)
HIVdata.head

# Anal(passive, vers) / Seks_alkohol (1.) / ShortPartnersAmount (not 1-10) (mix it with protection) /
# AnalProtec(sometimes, never) # Płeć (M)

HIVdata.dropna(subset=['Seks_alkohol'], inplace=True)
HIVdata = HIVdata[HIVdata['Płeć'] != 'I']
HIVdata = HIVdata[HIVdata['ShortPartnersAmount'] != 'na']
HIVdata['Anal'].unique()
HIVdata['ShortPartnersAmount'].unique()
HIVdata['Seks_alkohol'].unique()
HIVdata['Płeć'].unique()




conditionsHighRisk = (HIVdata['Seks_alkohol'] == 1.) & \
                     (HIVdata['Płeć'] == 'M') & \
                     (HIVdata['ShortPartnersAmount'] != '1-10') & \
                     ((HIVdata['Anal'] == 'passive') | (HIVdata['Anal'] == 'vers')) & \
                     ((HIVdata['AnalProtec'] == 'sometimes') | (HIVdata['AnalProtec'] == 'never'))

HIVdata.loc[:, 'HighRiskHIV'] = np.select([conditionsHighRisk], [1], default=[0])

HIVdata.groupby(['HighRiskHIV', 'Hetero_normative', 'HIV']).size().reset_index(name='Count')



# No gender HighRisk profile

conditionsHighRiskNoGender = (HIVdata['Seks_alkohol'] == 1.) & \
                     (HIVdata['ShortPartnersAmount'] != '1-10') & \
                     ((HIVdata['Anal'] == 'passive') | (HIVdata['Anal'] == 'vers')) & \
                     ((HIVdata['AnalProtec'] == 'sometimes') | (HIVdata['AnalProtec'] == 'never'))

HIVdata.loc[:, 'HighRiskHIVNoGen'] = np.select([conditionsHighRiskNoGender], [1], default=[0])

HIVdata.groupby(['HighRiskHIVNoGen', 'Hetero_normative', 'HIV']).size().reset_index(name='Count')





HIVdata.to_csv('PKD_modelHIGHRISK_DF.csv', index=False)
