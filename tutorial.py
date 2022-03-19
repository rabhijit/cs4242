from json import load
import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Test", page_icon=":tada:", layout="wide")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# -- Style --
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/styles.css")

# -- Loading assets --
lottie_coding = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_5wucvmas.json")

# -- Header --
with st.container():
    st.subheader("Hi! Tester here")
    st.title("A data analyst or sth")
    st.write("Bla bla yada yada")
    st.write("Yabba dabba doo!")

with st.container():
    str1 = ""
    user_input = st.text_input("label goes here", str1)
    st.write(str1)

# -- What I do
with st.container():
    st.write("---") # divider
    left_col, right_col = st.columns(2)
    with left_col:
        st.header("What I do is...")
        st.write("""Not much.""")

    with right_col:
        st_lottie(lottie_coding, height=300, key="coding")
        st.header("What I want to do is...")
        st.write("##")
        st.write("""Even less.""")

# -- Projects --
with st.container():
    st.write("---")
    st.header("Projects and stuff")
    st.write("##")
    image_col, text_col = st.columns((1, 1))

    with image_col:
        # l8r
        pass
    with text_col:
        st.subheader("WWEWEWE")
        st.write("""Something something""")
        st.markdown("What's this?")
        st.write("[Link example here](https://www.google.com)")
