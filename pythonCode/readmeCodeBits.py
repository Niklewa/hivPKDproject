# In this file you can find short description of the files in this folder

# DAG_sexualIdentity.py
# Various DAGs created for representation of relations between variables

# DataWranglinfForHomoCasualityCheck.py
# File in which I have done all the data wrangling of the initial PKD dataset
# Created many new variables: HIV status, risk profile, being hetoro-normative etc.

# MLog_DataHIGHRISKProfile.py
# Here I have build various logistic models basing on RP (risk profile) variable
# risk of HIV relative to gender and sexual indentity
# probability of being in a RP relative to gender and sexual identity

# MLog_homRIskProfiler.py
# Models for finding most risky behaviours, including a big model with all of the candidates
# categories such as: number of partners, sex preference, gender, use of protection, use of alcochol

# MLog_intercourseTypeComparison.py
# Logistic models for different type of sexual intercourse oral/anal and protection use

# MLog_SexAlcohol.py
# Logistic model testing if the alcochol induced intercourse increases the risk of infection (it does)

# MLogit_Partners_Heteronorm_gender.py
# Model testing what are the differnces between the influence of the number of partners relative
#  to gender and sexual identity

# MLogit_Partners.py
# testing the relation of the amount of partners and HIV infection

# MLogitBase_Heteronormativity.py
# building basic logistic models for testing the relation between being infected
# relative to gender and sexual identity

# Mmultinomial_RiskyCat.py
# A file in which I have tried to build multinomial (softmax) model that
# will calculate the probability for multiple categories 
# (having many sexual partners, enjoying certain type of sex etc.)

# PKD_vis_HeteroHomoMSM.py
# building some basic visualization that helped me to understand the dataset (EDA)