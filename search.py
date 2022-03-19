from json import load
from PIL import Image
import requests
import streamlit as st
from streamlit_lottie import st_lottie

#st.set_page_config(page_title="Test", page_icon=":tada:", layout="wide")
#st.session_state.search_input = ""

def view():
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
        #st_lottie(lottie_gif, height=220, key=2)
        st.markdown("<h6 style='text-align: center; margin: 0; padding-bottom: 0; font-weight:normal'>results for:</h6><h6 style='text-align: center; padding-top:0; font-size:40px'> r/" + st.session_state.search_input + "</h6>", unsafe_allow_html=True)
        #st.markdown("<h6 style='text-align: center; margin: 0; padding: 0; font-size:60px'>" + st.session_state.search_input + "</h6>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; font-weight:normal'>type in a subreddit below to browse its statistics!</h6>", unsafe_allow_html=True)

    with st.container():
        buff, search_col, buff2 = st.columns((1, 3, 1))
        with search_col:
            st.write("##")
            str1 = ""
            user_input = st.text_input("r/", str1, key=2)
            st.write(str1)

