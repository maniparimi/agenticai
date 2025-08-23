import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai


import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key = API_KEY)

# PAGE CONFIG

st.set_page_config(
    page_title = "Multimodel AI agent",
    page_icon = "*",
    layout = "wide"
)

st.title("Phidata Multimodal AI agent")
st.header("Powered by gemeni 2.0")

@st.cache_resource
def initisalise_agent():
    return Agent(
        name = "Video ai summaroser",
        model = Gemini(id = "gemini-2.0-flash-exp"),
        tools = [DuckDuckGo],
        markdown = True
    )

##initillise agent
multimodal_agent = initisalise_agent()

# file uploader
video_file = st.file_uploader(
    "Upload a video file", type= ['mp4', 'mov', 'avi'], help = "Upload a video for analysis"

)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    
    st.video(video_path, format= "video/mp4", start_time = 0)

    user_query = st.text_area(
        "what insights ate you seeking from the video?",
        placeholder = "ask anyhting",
        help = "Provide specific questions"
    )

    if st.button("analyse video", key="analyse video button"):
        if not user_query:
            st.warning("please enter question")
        else:
            try:
                with st.spinner("processing"):
                    #upload and process video
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    #prompt genereaiton 
                    analysis_prompt = (
                        f"""
                            anylise the uploaded video for content and context.
                            respond to the query using video insights 
                            {user_query}

                            provide a detailed response
                        """
                    )

                    # ai agent processing
                    response = multimodal_agent.run(analysis_prompt, videos= [processed_video])
                # display result
                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"an error:{error}")
            finally:
                # clean up temp video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("upload video file")

