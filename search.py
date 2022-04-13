from json import load
from PIL import Image
import requests
from config import NUMBER_OF_SUBREDDITS, NUMBER_OF_SUBMISSIONS_PER_SUBREDDIT
import streamlit as st
from streamlit_lottie import st_lottie

import reddit
import ml

USERS = 0
INFO = 1
NAMES = 2

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

    with st.container():
        buff, search_col, buff2 = st.columns((1, 3, 1))
        with search_col:
            st.write("##")
            try:
                with st.spinner("Loading statistics..."):
                    subreddit_name = reddit.get_real_subreddit_name(st.session_state.search_input, st.session_state.subreddit_data)
                    subreddit_description = reddit.get_subreddit_description(subreddit_name, st.session_state.subreddit_data)
                    subreddit_metrics = reddit.get_subreddit_metrics(subreddit_name, st.session_state.subreddit_data).style.hide_index()
                    interlinked_subreddits_by_user = reddit.get_interlinked_subreddits(subreddit_name, st.session_state.overlap_data)
                    interlinked_subreddits_by_algebra = ml.get_nearest_subreddit_vectors_by_user(subreddit_name, st.session_state.subreddit_data, st.session_state.vector_data)
                    subreddit_comments_wordcloud = ml.generate_wordcloud(subreddit_name, st.session_state.subreddit_data)
                    similar_subreddits_by_comment_tfidf = ml.get_nearest_subreddit_vectors_by_comment_tfidf(subreddit_name, st.session_state.comment_tfidf_vector_data)


                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Description</h6>", unsafe_allow_html=True)
                st.markdown("<h6 style='font-style:italic; font-weight:normal;'>" + subreddit_description + "</h6>", unsafe_allow_html=True)

                st.write("#")
                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Metrics</h6>", unsafe_allow_html=True)
                st.dataframe(subreddit_metrics)

                st.write("#")
                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Most related subreddits by comment TF-IDF angular similarity</h6>", unsafe_allow_html=True)
                st.table(similar_subreddits_by_comment_tfidf)

                st.write("#")
                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Wordcloud of the most common words used in this subreddit</h6>", unsafe_allow_html=True)
                st.pyplot(subreddit_comments_wordcloud)

                st.write("#")
                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Most related subreddits by user overlap percentage</h6>", unsafe_allow_html=True)
                st.table(interlinked_subreddits_by_user)
                
                st.write("#")
                st.markdown("<h6 style='font-size:20px; font-weight:bold;'>Most related subreddits by user angular similarity</h6>", unsafe_allow_html=True)
                st.table(interlinked_subreddits_by_algebra)

            except KeyError:
                st.warning(f"Sorry! The subreddit you have requested is not within the top {NUMBER_OF_SUBREDDITS} subreddits. Please try another subreddit.")


