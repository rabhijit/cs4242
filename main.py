from json import load
from json.tool import main
from PIL import Image
from config import NUMBER_OF_SUBREDDITS
import search
import requests
import streamlit as st
import hydralit_components as hc
from streamlit_lottie import st_lottie

import reddit
import ml

st.set_page_config(page_title="trenddit", page_icon=":tada:", layout="wide")
st.session_state.search_input = ""

with st.spinner("""
Loading data files...

(This will take a few minutes given the data size, and you might even get a warning from Streamlit about the "streamlit_lottie" component.
Rest assured that this is intended, since Streamlit will warn of a timeout after 100 seconds.)"""
):
    with st.spinner("Loading subreddit and user info..."):
        if 'subreddit_data' not in st.session_state:
            st.session_state.subreddit_data = reddit.load_subreddit_pickle()
        if 'overlap_data' not in st.session_state:
            st.session_state.overlap_data = reddit.load_overlap_pickle()
        if 'vector_data' not in st.session_state:
            st.session_state.vector_data = reddit.load_vector_pickle()
    with st.spinner("Retrieving comment data..."):
        if 'comment_tfidf_vector_data' not in st.session_state:
            st.session_state.comment_tfidf_vector_data = reddit.load_comments_tfidf_pickle()
        if 'wordcloud_data' not in st.session_state:
            st.session_state.wordcloud_data = reddit.load_wordcloud_pickle()

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
    st.write("##")
    st_lottie(lottie_gif, height=220, key=1)
    st.markdown("<h6 style='text-align: center; margin: 0; padding: 0; font-size:60px; padding-bottom:10px'>trenddit</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; font-weight:normal; padding-bottom:0'>type in a subreddit below to browse its statistics!</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; font-weight:normal'>results will be generated below.</h6>", unsafe_allow_html=True)
    st.markdown(f"<h6 style='text-align: center; font-weight:bold; font-style:italic'>note: statistics are only available for the top {NUMBER_OF_SUBREDDITS} SFW subreddits on Reddit.</h6>", unsafe_allow_html=True)

with st.container():
    buff, search_col, buff2 = st.columns((1, 3, 1))

    with search_col:
        st.write("##")
        st.session_state.search_input = st.text_input(label="r/", value="", key=1)
        
    buff, button_col, buff2 = st.columns((3, 3, 3))
    with button_col:
        st.write("#")
        if st.button("Or, click here to check out a cool visualization of subreddit relationships!"):
            with st.spinner("Creating subreddit map... just a moment!"):
                ml.plot_subreddit_clusters(reddit.load_vector_pickle())

# -- Search results --
if st.session_state.search_input != "":
    st.write("##")
    st.write("##")
    st.write("---")
    st.write("##")
    pages["search"].view()


