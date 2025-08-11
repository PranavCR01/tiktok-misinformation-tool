import os
import streamlit as st
import pandas as pd
import datetime
import time
import re
from openai import OpenAI, AzureOpenAI

# Import helper functions from processes
from pages.processes.api_helpers import is_open_ai_api_key_valid, get_model_selection, is_azure_api_key_valid
from pages.processes.transcription import transcribe, transcribe2, transcriber, split_video, display_split_files
from pages.processes.analysis import analyze, analyze2
from pages.processes.utils import remove_uploaded_files, convert_df, tokenizer, tokens_check, keyword

def main():
    st.title("Automatic Misinformation Analysis")
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "check_done" not in st.session_state:
        st.session_state["check_done"] = False
    if "service_provider" not in st.session_state:
        st.session_state["service_provider"] = False
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = False
    if "azure_api_key" not in st.session_state:
        st.session_state["azure_api_key"] = False
    if "azure_endpoint" not in st.session_state:
        st.session_state["azure_endpoint"] = False
    if "azure_api_version" not in st.session_state:
        st.session_state["azure_api_version"] = False
    if "azure_deployment_name" not in st.session_state:
        st.session_state["azure_deployment_name"] = False
    

    st.session_state["service_provider"] = st.selectbox("Select your LLM provider:", ("OpenAI", "Azure OpenAI"))

    if not st.session_state["service_provider"]:
        st.error("Please select a LLM provider")
    if st.session_state["service_provider"] == "OpenAI" and st.session_state["check_done"] == False:
        st.session_state["openai_api_key"] = st.text_input('Enter the OpenAI API key', type='password')
    if st.session_state["service_provider"] == "Azure OpenAI" and st.session_state["check_done"] == False:
        st.session_state["azure_api_key"] = st.text_input('Enter the Azure API key', type='password')
        st.session_state["azure_endpoint"] = st.text_input('Enter your Azure endpoint', type='password')
        st.session_state["azure_api_version"] = st.text_input('Enter your Azure api version', type='password')
        st.session_state["azure_deployment_name"] = st.text_input('Enter your Azure deployment name', type='password')

    if st.session_state["openai_api_key"]:
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        if is_open_ai_api_key_valid(st.session_state["openai_api_key"], client):
            st.success('OpenAI API key check passed')
            st.session_state["check_done"] = True
            client = OpenAI(api_key=st.session_state["openai_api_key"])
        else:
            st.error('OpenAI API key check failed. Please enter a correct API key')
    elif st.session_state["azure_api_key"] and st.session_state["azure_endpoint"] and st.session_state["azure_api_version"] and st.session_state["azure_deployment_name"]:
        client = AzureOpenAI(
            azure_endpoint = st.session_state["azure_endpoint"], 
            api_key= st.session_state["azure_api_key"],  
            api_version= st.session_state["azure_api_version"]
        )
        if is_azure_api_key_valid(st.session_state["azure_api_key"], client, st.session_state["azure_deployment_name"]):
            st.success('Azure OpenAI API initialization passed')
            st.session_state["check_done"] = True
            client = AzureOpenAI(
                azure_endpoint = st.session_state["azure_endpoint"], 
                api_key= st.session_state["azure_api_key"],  
                api_version= st.session_state["azure_api_version"]
            )
        else:
            st.error('Azure OpenAI initiliaztion failed. Please ensure you have filled all fields with the correct information.')
    
    if st.session_state["check_done"] and st.session_state["openai_api_key"]:
        selected_model = get_model_selection()

    video_files = st.file_uploader('Upload Your Video File', type=['wav', 'mp3', 'mp4'], accept_multiple_files=True)
    keywords_df = pd.DataFrame(columns=["Keyword", "Definition"])

    if st.button('Transcribe and Analyze Videos'):
        keyword_map = {}
        st.session_state.page_state = 'main'
        data = []
        keywords = ""
        container_array = []
        for i in range(1, len(video_files) + 1):
            container_name = f"container{i}"
            container_array.append(container_name)
        for i, video_file in enumerate(video_files):
            container_array[i] = st.status(f"Processing ({video_file.name})")
            time.sleep(0.7)
            with container_array[i] as container:
                if video_file is not None:
                    container.update(label=f"Transcribing ({video_file.name})...", state="running", expanded=False)
                    transcript = transcriber(video_file, client)
                    st.markdown(video_file.name + ": " + transcript)
                    container.update(label=f"Transcribed ({video_file.name})", state="running", expanded=True)
                    time.sleep(0.8)
                    container.update(label=f"Analyzing ({video_file.name})...", state="running", expanded=True)
                    if st.session_state["service_provider"] == "OpenAI":
                        analysis = analyze(transcript, selected_model, client, container)
                    elif st.session_state["service_provider"] == "Azure OpenAI":
                        analysis = analyze2(transcript, client, container, st.session_state["azure_deployment_name"])
                    st.markdown(analysis)
                    lines = analysis.splitlines()
                    for line in lines:
                        if "Classification Status:" in line:
                            misinformation_status = line.split("Classification Status:")[-1].strip()
                            break
                    for line in lines:
                        match = re.match(r"^\d+\.\s+(.*)", line)
                        if match:
                            clean_line = match.group(1)
                            parts = clean_line.split(":", 1)
                            if len(parts) == 2:
                                keyword = parts[0].strip()
                                definition = parts[1].strip()
                                keyword_map[keyword] = definition
                        else:
                            keywords = "Keywords not found."
                    d = {
                        "video_file": video_file.name,
                        "transcript": transcript, 
                        "misinformation_status": misinformation_status, 
                        "keywords": "", 
                    }
                    data.append(d)
                else:
                    st.sidebar.error("Please upload a video file")
                container.update(label=f'{video_file.name}', state="complete", expanded=False)
        st.session_state.data = data
        df = pd.DataFrame(data)
        # results = analyze_keywords_with_gpt(keyword_map, client, selected_model)
        # st.write(results)
        csv = convert_df(df)
        date_today_with_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        download_button_key = f"download_results_button_{st.session_state.page_state}_{time.time()}"
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name=f'results_{date_today_with_time}.csv',
            mime='text/plain',
            key=download_button_key)
    if st.button("Remove Files"):
        remove_uploaded_files()
        
if __name__ == "__main__":
    main()
