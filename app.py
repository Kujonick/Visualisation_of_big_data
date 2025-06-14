import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import os


MAX_EMBEDDINGS_ON_SCREEN = 20_000
REFINED_DATA_PATH = 'data\\refined.parquet'
EMBEDDINGS_FOLDER = "D:\\WDZD\\results"


# read the data
original_data = pd.read_parquet(REFINED_DATA_PATH)

# read embeddings
n_points = original_data.shape[0]


col1, col2 = st.columns(2)
with col1:
    method_name = st.radio("Choose method", [ "IncrementalPCA", "UMAP", "PACMAP", "PCA"])

with col2:
    dim = st.radio("Choose embedding dimensionality", ["2D", "3D"])



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
df['timestamp_numeric'] = df['timestamp'].astype('int64') / 1e9



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



if dim == "2D":
    fig = px.scatter(
        sampled_df, x='x', y='y',
        hover_data=['tweet'],
        color='timestamp_numeric',
        custom_data=['tweet'],
        title="Embedding 2D",
        color_continuous_scale='Turbo',
        range_color=[min_timestamp, max_timestamp] 
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
        color='timestamp_numeric',
        hover_data=['tweet'],
        custom_data=['tweet'],
        title="Embedding 3D",
        color_continuous_scale='Turbo',
        range_color=[min_timestamp, max_timestamp] 
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
fig.update_coloraxes(showscale=False)


selected = st.plotly_chart(fig, use_container_width=True)

