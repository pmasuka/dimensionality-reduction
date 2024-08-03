#!/usr/bin/env python
# coding: utf-8

# # Cluster countries

# In[ ]:


get_ipython().system('pip install umap')


# In[ ]:


get_ipython().system('pip install umap-learn')


# In[ ]:


get_ipython().system('pip install pandas matplotlib datashader bokeh holoviews scikit-image colorcet')


# Import the required modules.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import umap
import umap.umap_ as umap_
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, PanTool, ResetTool, BoxZoomTool
from bokeh.palettes import brewer, RdBu, Viridis
output_notebook()


# In[ ]:


from google.colab import files

# Upload the dataset file from your local drive to Google Colab
uploaded = files.upload()

# Once the file is uploaded, you can access it like this
for filename in uploaded.keys():
    print('Uploaded file "{name}" with length {length} bytes'.format(
        name=filename, length=len(uploaded[filename])))


# In[ ]:


import pandas as pd

# Assuming the uploaded file is a CSV file, you can read it into a pandas DataFrame
for filename in uploaded.keys():
    data_2015 = pd.read_csv(filename)
    break  # Assuming there's only one file uploaded, so we break after reading the first file


# ## Problem setting

# Let's try to cluster the countries baased on the indicators of happiness.

# ## Import data

# In[ ]:


# data_2015 = pd.read_csv('Data/world_happiness_2015.csv')


# In[ ]:


data_2015.head()


# In[ ]:


data_2015.describe()


# ## Preprocess data

# Define a class to extract data from a pandas DataFrame.

# In[ ]:


class FeatureSelector(BaseException, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    def fit(self, X, Y=None):
        return self
    def transform(self, X):
        return X[self._feature_names].values


# Define a transformer for the region names into one-hot encoding.

# In[ ]:


region_transformer = ColumnTransformer([('one_hot_encoder',
                                         OneHotEncoder(categories='auto'),
                                         ['Region'])])


# Define the names of the columns that hold numerical data.

# In[ ]:


num_attr_names = ['Economy (GDP per Capita)',
                  'Family', 'Health (Life Expectancy)', 'Freedom',
                  'Trust (Government Corruption)', 'Generosity',
                  'Dystopia Residual']


# Create a pipeline for the numerical attributes, rescaling them after selection.

# In[ ]:


num_attrs_pipeline = Pipeline([
    ('select_num_attrs', FeatureSelector(num_attr_names)),
    ('scaler', MinMaxScaler()),
])


# In[ ]:


preparation_pipeline = FeatureUnion(transformer_list=[
    ('region_attr', region_transformer),
    ('num_attrs', num_attrs_pipeline),
])


# Run the pipeline to prepare the data.

# In[ ]:


prepared_data = num_attrs_pipeline.fit_transform(data_2015)


# In[ ]:


prepared_data.shape


# ## Dimensionality reduction

# ### Principal Component Analysis (PCA)

# Perform a principal Component Analysis on the data.

# In[ ]:


pca = PCA()
pca_data = pd.DataFrame(pca.fit_transform(prepared_data))
pca_data.columns = [f'PC{i}' for i in range(1, 8)]


# We can check the variance ratio explained by each of the components.

# In[ ]:


pca.explained_variance_ratio_


# It is clear that most of the variance in the data is explained by the first two components, since this amounts to 65 %.  We can visualize the data in two dimension using the two principal components that explain most of the varianace.

# In order to assess the results, we want a plot that shows the country name, and the happiness score as well.  The country name can be added using a hover tool, the happiness score as the color of the glyphs.

# We add the country name and the happiness score to the PCA dataframe, as well as an additional column that encodes the score as a color from a red-blue Brewer color scheme.  The happiness score is converted to categorical data for this purpose.

# In[ ]:


pca_data['Country'] = data_2015['Country']
pca_data['Happiness Score'] = data_2015['Happiness Score']
happiness_min = data_2015['Happiness Score'].min()
happiness_max = data_2015['Happiness Score'].max()
nr_colors = int(happiness_max) - int(happiness_min) + 1
pca_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                     bins=np.linspace(np.floor(happiness_min),
                                                      np.ceil(happiness_max),
                                                      nr_colors + 1),
                                     labels=brewer['RdBu'][nr_colors])


# In[ ]:


pca_data


# Now we can create a plot with the PCA dataframe as source for the glyphs and their color, as well as the tooltip information that shows the country name.

# In[ ]:


hovertool = HoverTool(tooltips=[('Country', '@Country')])
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)
fig.circle('PC1', 'PC2', source=pca_data, fill_color='Happiness Color',
           size=8, alpha=0.85)
show(fig)


# It is clear from the plot above that the first principal component seems to be a good indicator for the countries with a high happiness score (colored red).

# ### Linear Discriminant Analysis (LDA)

# Linear Discriminant Analysis is a supervised learning technique that maximizes the between-category distances, while minimizing the within-category-distance.

# We create labels for the data by introducing categories based on the happiness score.

# In[ ]:


happiness = pd.DataFrame()
happiness['Country'] = data_2015.Country
happiness['Happiness Label'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['unhappy', 'neutral', 'happy'])


# Most countries are in the 'neutral' category, few in 'happy' or 'unhappy'.  Note that the label names are not normative.

# In[ ]:


happiness.groupby('Happiness Label').count()


# LDA is a supervised method, so we use the happiness label as output.

# In[ ]:


target = happiness['Happiness Label'].values


# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=2)
lda_data = pd.DataFrame(lda.fit(prepared_data, target) \
                           .transform(prepared_data))
lda_data.columns = ['C1', 'C2']


# The LDA produces two components for each country, and we can plot the countries using those similar as we did for the PCA.

# In[ ]:


lda_data['Country'] = data_2015['Country']
lda_data['Happiness Score'] = data_2015['Happiness Score']
lda_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                     bins=[0.0, 4.0, 7.0, 10.0],
                                     labels=['blue', 'yellow', 'red'])


# In[ ]:


hovertool = HoverTool(tooltips=[('Country', '@Country')])
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)
fig.circle('C1', 'C2', source=lda_data, fill_color='Happiness Color',
           size=8, alpha=0.85)
show(fig)


# ### t-SNE

# In[ ]:


tsne_data = pd.DataFrame(TSNE(learning_rate=70).fit_transform(prepared_data))
tsne_data.columns = ['C1', 'C2']


# In[ ]:


tsne_data['Country'] = data_2015['Country']
tsne_data['Happiness Score'] = data_2015['Happiness Score']
tsne_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])


# In[ ]:


tsne_data.head()


# In[ ]:


hovertool = HoverTool(tooltips=[('Country', '@Country')])
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)
fig.circle('C1', 'C2', source=tsne_data, fill_color='Happiness Color',
           size=8, alpha=0.85)
show(fig)


# ### UMAP

# In this section, we add dimensionality reduction using UMAP. We experiment with the following hyperparameters (n_neighbours, min_distance and metric) to see whether and how they influence the resulting visualization.

# ### Number of neighbours

# This parameter controls how UMAP balances local versus global structure in the data. It has a default value of 15. We will explore three iterations on n_neighbours: 2, 15 and 30

# In[ ]:


# Iteration 1: n_neighbors=2, min_dist=0.1, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=2, min_dist=0.1, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 2: n_neighbors=15, min_dist=0.1, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 3: n_neighbors=30, min_dist=0.1, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# The three iterations above for number of neighbours show clearly different patterns for each of the hyperparameter values. For n_neighbors=2, the points are very disconnected and scattered throughout the space, so no clear clusters of happiness levels can be seen.
# 
# As n_neighbors is increased to 15, a more clearer structure emerges, with high happiness cluster clearly separated from low and medium. The last two are intermingled a little bit. Overall, there is a fairly good overall view of the data showing how the various colors interelate to each other.
# 
# As n_neighbors increases to 30, much more focus in placed on the overall structure of the data. The overall structure is relatively well captured, although there is room for improvement especially for the red and yellow.

# ### Minimum distance

# The min_dist parameter provides the minimum distance apart that points are allowed to be in the low dimensional representation. The default value for min_dist is 0.1. We will iterate using the following discrete values: 0.0, 0.1 and 0.99.

# In[ ]:


# Iteration 4: n_neighbors=15, min_dist=0.0, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 5: n_neighbors=15, min_dist=0.1, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 6: n_neighbors=15, min_dist=0.99, n_components=2

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.99, n_components=2).fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# The three iterations above for minimum distance show clearly different patterns for each of the hyperparameter values. For min_dist=0.0, UMAP manages to find small connected components, with red and blue clusters at opposite ends and yellow cluster in the middle. As min_dist is increased these structures are pushed apart into softer more general features. The pattern for min_distance=0.99 spans the space more with less gaps between the clusters.

# ### Metric

# This controls how distance is computed in the ambient space of the input data. By default UMAP supports a wide variety of metrics, which can be grouped into Minkowski style metrics, Normalized spatial metrics and Angular and correlation metrics, among others. For the hyperparameter tuning, we selected one metric from each group, namely: Euclidean, mahalanobis and cosine respectively

# In[ ]:


# Iteration 7: n_neighbors=15, min_dist=0.1, n_components=2, metric = cosine

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine").fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 8: n_neighbors=15, min_dist=0.1, n_components=2, metric = mahalanobis

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="mahalanobis").fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# In[ ]:


# Iteration 9: n_neighbors=15, min_dist=0.1, n_components=2, metric = cosine

# Perform dimensionality reduction using UMAP
umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine").fit_transform(prepared_data)
umap_data = pd.DataFrame(umap_data, columns=['C1', 'C2'])

# Add 'Country', 'Happiness Score', and 'Happiness Color' columns to the UMAP transformed data
umap_data['Country'] = data_2015['Country']
umap_data['Happiness Score'] = data_2015['Happiness Score']
umap_data['Happiness Color'] = pd.cut(data_2015['Happiness Score'],
                                      bins=[0.0, 4.0, 7.0, 10.0],
                                      labels=['blue', 'yellow', 'red'])

# Create HoverTool
hovertool = HoverTool(tooltips=[('Country', '@Country'), ('Happiness Score', '@{Happiness Score}')])

# Create Bokeh plot
fig = figure(tools=[PanTool(), BoxZoomTool(), hovertool, ResetTool()],
             width=500, height=400)

# Plotting the UMAP transformed data
fig.circle('C1', 'C2', source=umap_data, fill_color='Happiness Color',
           size=9, alpha=0.7)

# Show the plot
show(fig)


# The above plots show the 2 dimensional representations of the happiness scores by countries. While the plots for Euclidean and cosine show some form of clustering, there seems to be no clear clustering. Of the three, mahalanobis shows three clearer clusters of blue and red at opposite ends and yellow in the middle. Thus, the choice of metric has a clear impact on the low dimensional representation.
