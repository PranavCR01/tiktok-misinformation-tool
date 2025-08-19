# pages/2_analysis.py

import os
import sys
import time
import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI, AzureOpenAI

# ---- Import app modules first; if not importable, patch sys.path then retry ----
try:
    from pages.processes.api_helpers import (
        is_open_ai_api_key_valid,
        get_model_selection,
        is_azure_api_key_valid,
        is_ollama_ready,
    )
    from pages.processes.transcription import transcriber
    from pages.processes.analysis import analyze, analyze2, analyze_local_mistral
    from pages.processes.utils import remove_uploaded_files, convert_df
except ModuleNotFoundError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from pages.processes.api_helpers import (
        is_open_ai_api_key_valid,
        get_model_selection,
        is_azure_api_key_valid,
        is_ollama_ready,
    )
    from pages.processes.transcription import transcriber  # type: ignore
    from pages.processes.analysis import (  # type: ignore
        analyze,
        analyze2,
        analyze_local_mistral,
    )
    from pages.processes.utils import remove_uploaded_files, convert_df  # type: ignore

# ---- Optional viz backends: Plotly preferred, else Altair ----
_PLOT_BACKEND = None
try:
    import plotly.express as px  # type: ignore

    _PLOT_BACKEND = "plotly"
except Exception:
    try:
        import altair as alt  # type: ignore

        _PLOT_BACKEND = "altair"
    except Exception:
        _PLOT_BACKEND = None


def _draw_label_pie(df: pd.DataFrame):
    if df.empty or "label" not in df.columns:
        return

    # Build a clean counts table with UNIQUE column names
    counts = (
        df["label"]
        .dropna()
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "label"})
        .astype({"count": "int64"})
    )

    if counts.empty:
        st.info("No labels to visualize yet.")
        return

    if _PLOT_BACKEND == "plotly":
        fig = px.pie(
            counts,
            names="label",
            values="count",
            title="Classification Breakdown",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif _PLOT_BACKEND == "altair":
        chart = (
            alt.Chart(counts)
            .mark_arc(innerRadius=70)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("label:N", legend=alt.Legend(title="Label")),
                tooltip=[alt.Tooltip("label:N"), alt.Tooltip("count:Q")],
            )
            .properties(title="Classification Breakdown")
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        st.info("Install plotly or altair to see the classification breakdown chart.")


def _draw_confidence_and_latency(df: pd.DataFrame):
    # --- Average confidence by label ---
    if "confidence_score" in df.columns and "label" in df.columns:
        conf_by_label = (
            df.dropna(subset=["confidence_score"])
            .groupby("label", as_index=False)["confidence_score"]
            .mean()
            .sort_values("label")
        )
        if not conf_by_label.empty:
            if _PLOT_BACKEND == "plotly":
                fig_conf = px.bar(
                    conf_by_label,
                    x="label",
                    y="confidence_score",
                    title="Average Confidence by Label",
                    labels={"confidence_score": "Avg. confidence", "label": "Label"},
                    range_y=[0, 1],
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            elif _PLOT_BACKEND == "altair":
                chart_conf = (
                    alt.Chart(conf_by_label)
                    .mark_bar()
                    .encode(
                        x="label:N",
                        y=alt.Y("confidence_score:Q", scale=alt.Scale(domain=[0, 1])),
                        tooltip=["label", "confidence_score"],
                    )
                    .properties(title="Average Confidence by Label")
                )
                st.altair_chart(chart_conf, use_container_width=True)

    # --- Model latency histogram ---
    if "time_taken_sec" in df.columns:
        lat = pd.to_numeric(df["time_taken_sec"], errors="coerce").dropna()
        if not lat.empty:
            if _PLOT_BACKEND == "plotly":
                fig_lat = px.histogram(
                    lat,
                    nbins=min(10, max(3, len(lat))),
                    title="Model Latency (seconds)",
                )
                fig_lat.update_layout(xaxis_title="Seconds", yaxis_title="Count")
                st.plotly_chart(fig_lat, use_container_width=True)
            elif _PLOT_BACKEND == "altair":
                lat_df = pd.DataFrame({"seconds": lat})
                chart_lat = (
                    alt.Chart(lat_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("seconds:Q", bin=alt.Bin(maxbins=10)),
                        y="count()",
                        tooltip=["count()"],
                    )
                    .properties(title="Model Latency (seconds)")
                )
                st.altair_chart(chart_lat, use_container_width=True)


def main():
    st.title("Automatic Misinformation Analysis")

    # ---- Session defaults ----
    defaults = {
        "api_key": "",
        "check_done": False,
        "service_provider": "",
        "openai_api_key": "",
        "azure_api_key": "",
        "azure_endpoint": "",
        "azure_api_version": "",
        "azure_deployment_name": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # ---- Provider select & credentials ----
    st.session_state["service_provider"] = st.selectbox(
        "Select your LLM provider:", ("OpenAI", "Azure OpenAI", "Ollama (Mistral)")
    )
    provider = st.session_state["service_provider"]

    client = None
    selected_model = None

    if provider == "OpenAI" and not st.session_state["check_done"]:
        st.session_state["openai_api_key"] = st.text_input(
            "Enter the OpenAI API key", type="password"
        )

    if provider == "Azure OpenAI" and not st.session_state["check_done"]:
        st.session_state["azure_api_key"] = st.text_input(
            "Enter the Azure API key", type="password"
        )
        st.session_state["azure_endpoint"] = st.text_input(
            "Enter your Azure endpoint", type="password"
        )
        st.session_state["azure_api_version"] = st.text_input(
            "Enter your Azure api version", type="password"
        )
        st.session_state["azure_deployment_name"] = st.text_input(
            "Enter your Azure deployment name", type="password"
        )

    if provider == "OpenAI" and st.session_state["openai_api_key"]:
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        if is_open_ai_api_key_valid(st.session_state["openai_api_key"], client):
            st.success("OpenAI API key check passed")
            st.session_state["check_done"] = True
            selected_model = get_model_selection()
        else:
            st.error("OpenAI API key check failed. Please enter a correct API key")

    elif provider == "Azure OpenAI" and all(
        st.session_state[k]
        for k in (
            "azure_api_key",
            "azure_endpoint",
            "azure_api_version",
            "azure_deployment_name",
        )
    ):
        client = AzureOpenAI(
            azure_endpoint=st.session_state["azure_endpoint"],
            api_key=st.session_state["azure_api_key"],
            api_version=st.session_state["azure_api_version"],
        )
        if is_azure_api_key_valid(
            st.session_state["azure_api_key"],
            client,
            st.session_state["azure_deployment_name"],
        ):
            st.success("Azure OpenAI API initialization passed")
            st.session_state["check_done"] = True
        else:
            st.error("Azure OpenAI initialization failed. Please verify credentials.")

    elif provider == "Ollama (Mistral)":
        if is_ollama_ready():
            st.success("Ollama detected locally. No API key required.")
            st.session_state["check_done"] = True
        else:
            st.error(
                "Ollama not reachable at http://localhost:11434. "
                "Install/run Ollama and `ollama pull mistral`."
            )

    # ---- Uploader ----
    video_files = st.file_uploader(
        "Upload Your Video File", type=["wav", "mp3", "mp4"], accept_multiple_files=True
    )

    # ---- Run pipeline ----
    if st.button("Transcribe and Analyze Videos"):
        if not video_files:
            st.sidebar.error("Please upload at least one video file")
            return
        if provider in ("OpenAI", "Azure OpenAI") and not client:
            st.sidebar.error("Please configure your provider/API key first")
            return
        if provider == "OpenAI" and not selected_model:
            st.sidebar.error("Please select an OpenAI model")
            return

        rows = []
        for video_file in video_files:
            with st.status(f"Processing ({video_file.name})") as container:
                # Transcribe
                container.update(
                    label=f"Transcribing ({video_file.name})…",
                    state="running",
                    expanded=False,
                )
                transcript = transcriber(video_file, client)
                container.update(
                    label=f"Transcribed ({video_file.name})",
                    state="running",
                    expanded=True,
                )

                # Analyze
                container.update(
                    label=f"Analyzing ({video_file.name})…",
                    state="running",
                    expanded=True,
                )
                if provider == "OpenAI":
                    result = analyze(transcript, selected_model, client, container)
                elif provider == "Azure OpenAI":
                    result = analyze2(
                        transcript,
                        client,
                        container,
                        st.session_state["azure_deployment_name"],
                    )
                else:
                    result = analyze_local_mistral(
                        transcript, container, model_name="mistral"
                    )

                rows.append(
                    {
                        "video_file": video_file.name,
                        "transcript": transcript,
                        "label": result.get("label", "CANNOT_RECOGNIZE"),
                        "keywords": ";".join(result.get("keywords", [])),
                        "confidence_score": round(
                            float(result.get("confidence", 0.0)), 2
                        ),
                        "time_taken_sec": result.get("time_taken_secs"),
                    }
                )
                container.update(
                    label=f"{video_file.name}", state="complete", expanded=False
                )

        df = pd.DataFrame(rows)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("Visualizations")
        _draw_label_pie(df)
        _draw_confidence_and_latency(df)

        csv = convert_df(df)
        stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name=f"results_{stamp}.csv",
            mime="text/csv",
            key=f"download_results_{time.time()}",
        )

    if st.button("Remove Files"):
        remove_uploaded_files()


if __name__ == "__main__":
    main()
