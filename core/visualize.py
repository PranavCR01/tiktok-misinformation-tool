import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import pandas as pd
import streamlit as st

def label_bar(df: pd.DataFrame):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots()
    counts.plot.bar(ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Classification Breakdown")
    st.pyplot(fig)

def keyword_network(df: pd.DataFrame, top_n: int = 10):
    all_kw = [kw for kws in df["keywords"] for kw in kws]
    common = Counter(all_kw).most_common(top_n)
    G = nx.Graph()
    for kw, freq in common:
        G.add_node(kw, size=freq)
    fig, ax = plt.subplots(figsize=(6,6))
    nx.draw(G, with_labels=True, node_size=[v*100 for _,v in common], ax=ax)
    st.pyplot(fig)
