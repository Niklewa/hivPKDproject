from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import seaborn as sns


import jax.numpy as jnp
from jax import random, vmap

# Loading the data

PKD_model_DF = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKD_model_DF.csv")

PKD_model_DF.columns

pd.set_option('display.max_columns', None)
PKD_model_DF.head

##########################################

Anal_HIV_Prot = PKD_model_DF.groupby(['Anal', 'AnalProtec', 'HIV']).size().reset_index(name='Count')



#
df_filtered = Anal_HIV_Prot[Anal_HIV_Prot['AnalProtec'] != 'noAnal']

# Create the bar plot with subplots and HIV category
g = sns.FacetGrid(df_filtered, col='AnalProtec', hue='HIV', sharex=False, sharey=False, col_wrap=2, height=3, aspect=1.5)
g.map_dataframe(sns.barplot, x='Anal', y='Count', ci=None)
g.set_axis_labels('Anal', 'Count')

g.add_legend(title='HIV', labels=['HIV=0', 'HIV=1'])

# Determine the maximum count value across all subplots
max_count = df_filtered['Count'].max()

# Set a common y-axis limit for all subplots
g.set(ylim=(0, max_count))

plt.tight_layout()
plt.show()



# Calculate percentages for each bar
df_filtered['Percentage'] = df_filtered['Count'] / df_filtered['AnalProtec'].map(total_counts) * 100

# Create the visualization
g = sns.FacetGrid(df_filtered, col='AnalProtec', hue='HIV', sharex=False, sharey=False, col_wrap=2, height=3, aspect=1.5)
g.map_dataframe(sns.barplot, x='Anal', y='Percentage', ci=None)
g.set_axis_labels('Anal', 'Percentage')

# Set the y-axis limit based on the maximum percentage
max_percentage = df_filtered['Percentage'].max()
g.set(ylim=(0, max_percentage))

# Display percentages on top of the bars
for ax in g.axes.flat:
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center')

# Show the plot
plt.show()