#core/visualize.py
import pandas as pd
import plotly.express as px
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st

def label_bar(df: pd.DataFrame):
    fig = px.pie(df, names="label", title="Classification Breakdown", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

def keyword_network(df: pd.DataFrame, top_n: int = 10):
    all_kw = []
    for kws in df["keywords"]:
        if isinstance(kws, str):
            all_kw.extend(k.strip() for k in kws.split(";") if k.strip())
        elif isinstance(kws, list):
            all_kw.extend(kws)

    common = Counter(all_kw).most_common(top_n)
    G = nx.Graph()
    for kw, freq in common:
        G.add_node(kw, size=freq)

    fig, ax = plt.subplots(figsize=(4, 4))
    pos = nx.spring_layout(G, k=0.6, iterations=20)
    node_sizes = [v * 100 for _, v in common]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, font_size=8, ax=ax)
    st.pyplot(fig)
