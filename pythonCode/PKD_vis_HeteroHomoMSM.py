from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go



HIVdata = pd.read_csv("C:\\Users\\nikod\\Documents\\RProjects\\hivProject\\PKDjoint.csv")

HIVdata.columns

pd.set_option('display.max_columns', None)
HIVdata.head


HIVdata["Powód"].unique()
# man sex with man: homo, bisek, MSM 

HIVdata.dropna(subset=["Powód"], inplace=True)

msm_mask = HIVdata["Powód"].str.contains(r"\bhomo\b|\bbisek\b|\bMSM\b", case=False, regex=True)

HIVdata["MSM_cause"] = msm_mask.astype(int)

HIVdata["MSM_cause"].unique()

# now a column with non heteroNorm ####################
HIVdata["Orientacja"].unique()

HIVdata.dropna(subset=["Orientacja"], inplace=True)


# Define a custom function to use with apply()
def classify_orientation(orientation):
    if orientation in ["homoseksualna", "biseksualna"]:
        return 1
    elif orientation == "heteroseksualna":
        return 0
    else:
        return 2

# Apply the custom function to each row of the "Orientacja" column, and save the results in a new column called "non_hetero"
HIVdata["non_hetero"] = HIVdata["Orientacja"].apply(classify_orientation)


############# VIS
grouped = HIVdata.groupby(["non_hetero", "MSM_cause"]).size()

# Reshape the grouped data into a pivot table
pivoted = grouped.unstack()

# Reorder the columns so that the "non_hetero" values are at the bottom
pivoted = pivoted[[0,1]]


labels = ["heteroseksualna", "homoseksualna", "inna"]

x_positions = np.arange(len(labels))

# Set the width of each bar
bar_width = 0.35

# Plot the bars for each "non_hetero" category side by side
fig, ax = plt.subplots()
ax.bar(x_positions - bar_width/2, pivoted[0], width=bar_width, label="Not MSM")
ax.bar(x_positions + bar_width/2, pivoted[1], width=bar_width, label="MSM")

# Add x-axis tick labels, title, and legend
ax.set_xticks(x_positions)
ax.set_xticklabels(labels)
ax.set_title("Counts of MSM_cause by non_hetero")
ax.legend()
plt.show()





fig = go.Figure()

fig.add_trace(go.Bar(x=pivoted.index, y=pivoted[0], name='MSM_cause_0'))
fig.add_trace(go.Bar(x=pivoted.index, y=pivoted[1], name='MSM_cause_1'))

fig.update_layout(
    title='MSM_cause by non_hetero',
    xaxis_title='non_hetero',
    yaxis_title='count',
    barmode='group',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

fig.show()


