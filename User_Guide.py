# pages/user_guide.py

import streamlit as st


def main():
    st.title("User Guide")

    st.markdown(
        """
    ## Introduction
    Welcome to the **Automatic Misinformation Analysis** tool. This application allows you to upload video files, transcribe their content, and analyze the transcriptions for potential misinformation.
    """
    )

    st.markdown(
        """
    ## How to Use
    """
    )

    st.markdown(
        """
    ### Step 1: Enter OpenAI API Key
    - On the "Analysis" page, you will be prompted to enter your OpenAI API key. This key is required to access the OpenAI services for transcription and analysis.
    - Enter your API key in the provided input box and click **Submit**.
    """
    )

    st.markdown(
        """
    ### Step 2: Upload Video Files
    - After successfully entering your API key, you can upload your video files.
    - Click on the **Upload Your Video File** button and select the video files you want to analyze. You can upload multiple files at once.
    """
    )

    st.markdown(
        """
    ### Step 3: Transcribe and Analyze
    - Once your video files are uploaded, click on the **Transcribe and Analyze Videos** button.
    - The application will transcribe the content of the videos and analyze them for potential misinformation.
    - The results will be displayed on the screen, including the transcription, keywords, and misinformation status.
    """
    )

    st.markdown(
        """
    ### Step 4: Data Visualization
    - After the videos are transcribed, you can access the Data Visualization component.
    - Navigate to the **Data Visualization** page from the sidebar.
    - This section provides visual representations of the analysis results, including:
        - A bar chart showing the frequency of misinformation vs. non-misinformation videos.
        - A network graph depicting the relationships between the top 10 frequent words in the video transcripts.
    - These visualizations help in better understanding the analysis results.
    """
    )

    st.markdown(
        """
    ### Step 5: Download Results
    - After the analysis is complete, you can download the results as a CSV file.
    - Click on the **Download results as CSV** button to save the results to your local machine.
    """
    )

    st.markdown(
        """
    ### Step 6: Remove Uploaded Files
    - If you want to remove the uploaded files from the session, click on the **Remove Files** button.
    """
    )

    st.markdown(
        """
    ## Contact
    If you have any questions or need further assistance, please contact [ianb@illinois.edu](mailto:ianb@illinois.edu).
    """
    )


if __name__ == "__main__":
    main()
