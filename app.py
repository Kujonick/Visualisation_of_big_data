import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import os
import json


MAX_EMBEDDINGS_ON_SCREEN = 20_000
REFINED_DATA_PATH = 'data'



# read the data
dataset_name = st.radio("Choose dataset", ["Sentiment", "Data Science"])
if dataset_name == "Sentiment":
    original_data = pd.read_parquet(os.path.join(REFINED_DATA_PATH, 'refined.parquet'))
    EMBEDDINGS_FOLDER = "D:\\WDZD\\results\\sentiment"
else:
    original_data = pd.read_parquet(os.path.join(REFINED_DATA_PATH, 'refined_ds.parquet'))
    original_data['field'] = original_data['field'].astype(str)
    EMBEDDINGS_FOLDER = "D:\\WDZD\\results\\data-science"
    with open('data\\field_mapping.json', "r", encoding="utf-8") as f:
        field_mapping = json.load(f)
        original_data['field'] = original_data['field'].map(field_mapping)

original_data['topic'] = original_data['topic'].astype(str)


with open('data\\topic_mapping.json', "r", encoding="utf-8") as f:
    class_mapping = json.load(f)
original_data['topic'] = original_data['topic'].map(class_mapping)

# read embeddings
n_points = original_data.shape[0]


col1, col2, col3= st.columns(3)
with col1:
    method_name = st.radio("Choose method", [ "IncrementalPCA", "UMAP", "PACMAP", "PCA"])

with col2:
    dim = st.radio("Choose embedding dimensionality", ["2D", "3D"])

with col3:
    labels =  ["time", "topic"]
    if dataset_name == "Data Science":
        labels.append('field')
    labeling = st.radio("Choose labeling",labels)


match method_name:
    case "UMAP":
        embedding_file = "umap_result"
    case "IncrementalPCA":
        embedding_file = "incremental_pca_result"
    case 'PCA':
        embedding_file = "pca_result"
    case 'PACMAP':
        embedding_file = "pacmap_result"
        
    
    case _:
        raise Exception("What the hell just did happen")
    
if dim == '3D':
    embedding_file = embedding_file + '_3d'
    
embeddings = np.load(os.path.join(EMBEDDINGS_FOLDER, embedding_file + '.npy'), allow_pickle=True)

if dim == '3D':
    embeddings_df = pd.DataFrame(embeddings.reshape(n_points, 3), columns=['x', 'y', 'z'])
else:
    embeddings_df = pd.DataFrame(embeddings.reshape(n_points, 2), columns=['x', 'y'])


df = original_data.join(embeddings_df)
df['timestamp'] = pd.to_datetime(df['date'])
df['timestamp_numeric'] = df['timestamp'].astype('int64') / (1e9 if dataset_name == 'Sentiment' else 1e6)


min_date = df['timestamp'].min().to_pydatetime()
max_date = df['timestamp'].max().to_pydatetime()

min_timestamp = int(min_date.timestamp())
max_timestamp = int(max_date.timestamp())


start_date, end_date = st.slider(
    "Data range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD hh-mm-ss",
    step=datetime.timedelta(hours=6)
)

filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
if filtered_df.shape[0] > MAX_EMBEDDINGS_ON_SCREEN:
    sampled_df = filtered_df.sample(n=MAX_EMBEDDINGS_ON_SCREEN)
else:
    sampled_df = filtered_df

if labeling == 'time':
    labeling_kwargs = dict(
        color='timestamp_numeric',
        color_continuous_scale='Turbo',
        range_color=[min_timestamp, max_timestamp] 
    )
else:
    color_map = {topic: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, topic in enumerate(df[labeling].unique())}
    labeling_kwargs = dict(
        color=labeling,
        color_discrete_map=color_map
    )



if dim == "2D":
    fig = px.scatter(
        sampled_df, x='x', y='y',
        hover_data=['tweet'],
        custom_data=['tweet'],
        title="Embedding 2D",
        **labeling_kwargs
    )
    fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    marker=dict(size=6)
    )

    fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
    )
else:
    fig = px.scatter_3d(
        sampled_df, x='x', y='y', z='z',
        hover_data=['tweet'],
        custom_data=['tweet'],
        title="Embedding 3D",
        **labeling_kwargs
    )
    fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    marker=dict(size=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            ),
            yaxis=dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            ),
            zaxis=dict(
                visible=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

if labeling == 'time':
    fig.update_coloraxes(showscale=False)
else:
    fig.update_layout(legend = dict(itemsizing='constant'))


selected = st.plotly_chart(fig, use_container_width=True)

