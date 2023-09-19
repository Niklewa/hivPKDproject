# Bayesian Modeling of HIV Risk Factors: Confronting HIV as a Stereotypical Infection of Sexual Minorities

This study challenges the stereotype associating HIV infection primarily with non-heteronormative males. Through Bayesian logistic regression models and risk profile analysis, I have uncovered that risky behaviors, rather than sexual identity or gender, are the primary drivers of HIV infection. 
The research employs a dataset created from a survey conducted in Polish HIV diagnostic centers. Poland is currently experiencing a rise in new HIV infections, with non-heteronormative males constituting the majority ofnew cases. However, the findings reveal that high-risk behavior serves asa more accurate predictor of infection than gender or sexual identity. This highlights the complexity of the factors driving new infections among non-heteronormative males.
This research underscores the critical need for comprehensive awarenessand testing to combat the ongoing HIV epidemic. It prompts a reevalua-tion of the assumptions surrounding the causes of the domination of non-heteronormative males in new infections

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)


## Project Overview

The main part of this project is a paper (HIVPaper.pdf) that can be found in the paper folder. There, you can find a detailed description with appropriate visualizations of my research. All the code that the paper is based on can be found in the pythonCode folder, along with a file readmeCodeBits.py that provides a short description of its contents.

The dataset is located in the dataBits folder, with the initial file, PKDjoint.csv, before any modifications. The acquisition of this file was made possible through the cooperation with PKD officials, to whom I am grateful.

I have utilized the Python 'numpyro' package to build Bayesian logistic regression models for calculating the probability of HIV infection for specific categories. The initial model indicated that non-heterosexual males have the highest probability of getting infected. However, in further research, I discovered better predictors. Using these predictors, such as the number of sexual partners, risky anal sex, and alcohol-induced sex, I created the RP (risk profile), which showed a higher probability of infection than non-heterosexual males.

I argue that we should not attribute gender and sexual identity as the primary causes of HIV infection. Instead, it is certain risky behaviors that act as the genuine causes.

## Prerequisites

The package requirements can be found in requirements.txt. The paper was written using Quarto, a program for research writing in markdown format. My Quarto setup uses the XeTeX latex engine, version 3.141592653-2.6-0.999995 (TeX Live 2023) (preloaded format=xelatex).

Some of the files in this repository are large, so I am using LFS (Large File System) to handle them. The files managed by this program are listed in the .gitattributes file.

