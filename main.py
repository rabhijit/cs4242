from json import load
from json.tool import main
from PIL import Image
import search
import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="trenddit", page_icon=":tada:", layout="wide")
st.session_state.search_input = ""

pages = {
    "main": main,
    "search": search
}

# -- Style --
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/styles.css")

# -- Loading assets --
lottie_gif = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_5wucvmas.json")

# -- Body --
with st.container():
    st_lottie(lottie_gif, height=220, key=1)
    st.markdown("<h6 style='text-align: center; margin: 0; padding: 0; font-size:60px; padding-bottom:10px'>trenddit</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; font-weight:normal; padding-bottom:0'>type in a subreddit below to browse its statistics!</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; font-weight:normal'>results will be generated below.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; font-weight:bold; font-style:italic'>note: statistics are only available for the top 3000 subreddits on Reddit.</h6>", unsafe_allow_html=True)

with st.container():
    buff, search_col, buff2 = st.columns((1, 3, 1))
    with search_col:
        st.write("##")
        st.session_state.search_input = st.text_input(label="r/", value="", key=1)

# -- Search results --
if st.session_state.search_input != "":
    st.write("##")
    st.write("##")
    st.write("---")
    st.write("##")
    pages["search"].view()


