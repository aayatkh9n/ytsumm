import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard

class AIVideoSummarizer:
    def __init__(self):
        load_dotenv()
        self.setup_apis()

    def setup_apis(self):
        self.model_env_checker = []
        if os.getenv("GOOGLE_GEMINI_API_KEY"):
            self.model_env_checker.append("Gemini")
            genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
        if os.getenv("OPENAI_CHATGPT_API_KEY"):
            self.model_env_checker.append("ChatGPT")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))

    def get_video_id(self, url):
        # Extract video ID from YouTube URL
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        return None

    def get_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([t['text'] for t in transcript_list])
        except Exception as e:
            st.error(f"Error getting transcript: {str(e)}")
            return None

    def get_timestamp_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            formatted_transcript = []
            for t in transcript_list:
                timestamp = int(t['start'])
                minutes = timestamp // 60
                seconds = timestamp % 60
                formatted_transcript.append(f"[{minutes}:{seconds:02d}] {t['text']}")
            return "\n".join(formatted_transcript)
        except Exception as e:
            st.error(f"Error getting transcript: {str(e)}")
            return None

    def generate_summary_gemini(self, transcript):
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Summarize this video transcript concisely:\n\n{transcript}"
        response = model.generate_content(prompt)
        return response.text

    def generate_summary_chatgpt(self, transcript):
        prompt = f"Summarize this video transcript concisely:\n\n{transcript}"
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_timestamps_gemini(self, transcript, video_url):
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Create timestamps for key moments in this video transcript. Format as '00:00 - Topic'. Video URL: {video_url}\n\n{transcript}"
        response = model.generate_content(prompt)
        return response.text

    def generate_timestamps_chatgpt(self, transcript, video_url):
        prompt = f"Create timestamps for key moments in this video transcript. Format as '00:00 - Topic'. Video URL: {video_url}\n\n{transcript}"
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def run(self):
        st.set_page_config(page_title="AI Video Summarizer", page_icon="🎥", layout="wide")
        st.title("AI Video Summarizer")

        if not self.model_env_checker:
            st.warning('No API keys found in environment. Please add API keys.', icon="⚠️")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            youtube_url = st.text_input("Enter YouTube Video Link")
            if youtube_url:
                video_id = self.get_video_id(youtube_url)
                if video_id:
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                else:
                    st.error("Invalid YouTube URL")
                    return

        with col2:
            if self.model_env_checker:
                self.model_name = st.selectbox('Select the model', self.model_env_checker)
                if self.model_name == "Gemini":
                    st.columns(3)[1].image("https://i.imgur.com/w9izNH5.png", use_column_width=True)
                elif self.model_name == "ChatGPT":
                    st.columns(3)[1].image("https://i.imgur.com/Sr9e9ZC.png", use_column_width=True)

        with col3:
            mode = st.radio(
                "What would you like to generate?",
                ["📝 Summary", "⏰ Timestamps", "📄 Transcript"],
                index=0
            )

        if youtube_url and video_id:
            if mode == "📝 Summary":
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        transcript = self.get_transcript(video_id)
                        if transcript:
                            if self.model_name == "Gemini":
                                summary = self.generate_summary_gemini(transcript)
                            else:
                                summary = self.generate_summary_chatgpt(transcript)
                            st.markdown("## Summary:")
                            st.write(summary)
                            st_copy_to_clipboard(summary)

            elif mode == "⏰ Timestamps":
                if st.button("Generate Timestamps"):
                    with st.spinner("Generating timestamps..."):
                        transcript = self.get_timestamp_transcript(video_id)
                        if transcript:
                            if self.model_name == "Gemini":
                                timestamps = self.generate_timestamps_gemini(transcript, youtube_url)
                            else:
                                timestamps = self.generate_timestamps_chatgpt(transcript, youtube_url)
                            st.markdown("## Timestamps:")
                            st.markdown(timestamps)
                            st_copy_to_clipboard(timestamps)

            else:  # Transcript mode
                if st.button("Get Transcript"):
                    with st.spinner("Fetching transcript..."):
                        transcript = self.get_transcript(video_id)
                        if transcript:
                            st.markdown("## Transcript:")
                            st.write(transcript)
                            st_copy_to_clipboard(transcript)
                            st.download_button(
                                label="Download transcript",
                                data=transcript,
                                file_name="transcript.txt"
                            )

        # Footer
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <p>Made with ❤️ by Aayat</a></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = AIVideoSummarizer()
    app.run()
