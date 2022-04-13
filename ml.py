from cmath import pi
import pickle
import logging

import pandas as pd
import numpy as np
import scipy.sparse as ss
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import reddit
from config import *


logger = logging.getLogger()
logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s: |%(name)s| %(message)s")


"""
Features to implement:
- Calculating similarity scores based on word association?
    - Requires tokenization, removal of stopwords etc
    - Search for a subreddit, display all the similar subs by word association

"""


def load_subreddit_vectors(overlap_data):
    subreddit_popularity = overlap_data.groupby('s2')['overlap'].sum()
    subreddits = np.array(subreddit_popularity.sort_values(ascending=False).index)
    index_map = dict(np.vstack([subreddits, np.arange(subreddits.shape[0])]).T)
    count_matrix = ss.coo_matrix((overlap_data.overlap,
                                (overlap_data.s2.map(index_map),
                                overlap_data.s1.map(index_map))),
                                shape=(subreddits.shape[0], subreddits.shape[0]),
                                dtype=np.float64)

    conditional_prob_matrix = count_matrix.tocsr()
    conditional_prob_matrix = normalize(conditional_prob_matrix, norm='l1', copy=False)
    reduced_vectors = TruncatedSVD(n_components=NUMBER_OF_SUBREDDITS - 2, random_state=1).fit_transform(conditional_prob_matrix)
    reduced_vectors = normalize(reduced_vectors, norm='l2', copy=False)

    subreddit_popularity = overlap_data.groupby('s2')['overlap'].sum()
    subreddits = np.array(subreddit_popularity.sort_values(ascending=False).index)
    seed_state = np.random.RandomState(0)
    subreddit_map = TSNE(perplexity=50.0, random_state=seed_state).fit_transform(reduced_vectors)
    subreddit_map_df = pd.DataFrame(subreddit_map, columns=('x', 'y'))
    subreddit_map_df['subreddit'] = subreddits

    save_to_pickle(subreddit_map_df, VECTORS_PKL)

    return subreddit_map_df


def load_subreddit_comment_tfidf_vectors(subreddit_data):
    combined_comments = {}
    for subreddit_name in subreddit_data[COMMENTS]:
        all_comments = []
        for comment in subreddit_data[COMMENTS][subreddit_name]:
            comment = comment.lower()
            comment_words = comment.split(" ")
            comment_words = [word for word in comment_words if word not in STOPWORDS]
            all_comments.append(" ".join(comment_words))
        combined_comments[subreddit_name] = " ".join(all_comments)

    df_comments = pd.DataFrame.from_dict(combined_comments, orient='index')
    df_comments = df_comments.reset_index()
    df_comments = df_comments.rename(columns = {'index': 'subreddit_name', 0: "combined_comments"})

    vectorizer = TfidfVectorizer(max_features=256)
    tfidf_vectors = vectorizer.fit_transform(df_comments['combined_comments'])

    subreddit_names = df_comments['subreddit_name'].tolist()
    tfidf_vectors = tfidf_vectors.toarray()

    comment_tfidf_vectors = dict(zip(subreddit_names, tfidf_vectors))

    save_to_pickle(comment_tfidf_vectors, COMMENT_TFIDF_VECTORS_PKL)


def get_nearest_subreddit_vectors_by_user(subreddit_name, vector_data):
    subreddit_coords = vector_data.loc[vector_data['subreddit'] == subreddit_name]
    subreddit_coords = np.array([subreddit_coords['x'], subreddit_coords['y']]).reshape(1, -1)
    vector_data = vector_data[vector_data['subreddit'] != subreddit_name]
    vector_data['distance'] = vector_data.apply(lambda row: cdist(subreddit_coords, np.array([row.x, row.y]).reshape(1, -1))[0][0], axis=1)
    vector_data = vector_data.drop(['x', 'y'], axis='columns')
    vector_data.set_index(keys='subreddit', inplace=True)
    return vector_data['distance'].nsmallest(n=10)

    '''
    subreddit_names = pd.DataFrame(subreddit_data[NAMES].items())
    subreddit_index = subreddit_names.index[subreddit_names[0] == subreddit_name.lower()].tolist()[0]
    this_subreddit_vector = vector_data[subreddit_index]

    angular_scores = []
    for i in range(len(vector_data)):
        subreddit_vector = vector_data[i]
        if i != subreddit_index:
            cos_sim = cosine_similarity([this_subreddit_vector], [subreddit_vector])[0][0]
            ang_sim = 1 - np.arccos(cos_sim) / pi
            angular_scores.append(ang_sim)
        else:
            angular_scores.append(-1)

    top_ten_subreddits = list(np.array(angular_scores).argsort()[-10:][::-1])
    top_ten_values = [angular_scores[idx] for idx in top_ten_subreddits]
    for i in range(0, len(top_ten_subreddits)):
        top_ten_subreddits[i] = subreddit_names.iloc[top_ten_subreddits[i]][1]

    top_ten_subreddits = dict(zip(top_ten_subreddits, top_ten_values))
    df = pd.DataFrame(top_ten_subreddits.items())
    df.columns = ['subreddit', 'angular similarity']
    df.set_index(keys='subreddit', inplace=True)

    return df
    '''


def get_nearest_subreddit_vectors_by_comment_tfidf(subreddit_name, comment_tfidfs):
    tfidf_similarities = {}

    for other_subreddit_name, tfidf_vector in comment_tfidfs.items():
        if other_subreddit_name == subreddit_name:
            continue

        tfidf_similarities[other_subreddit_name] = cosine_similarity(comment_tfidfs[subreddit_name].reshape(1, -1), tfidf_vector.reshape(1, -1))[0]
    
    similarity_scores = pd.DataFrame.from_dict(tfidf_similarities, orient='index')
    similarity_scores = similarity_scores.rename(columns = {0: "tfidf_similarity"})
    similarity_scores = similarity_scores.sort_values(by="tfidf_similarity", ascending=False)
    return similarity_scores.head(10)
        

def plot_subreddit_clusters(vector_data):
    kmeans = KMeans(n_clusters=25, random_state=0)
    vector_data['cluster'] = kmeans.fit_predict(vector_data[['x', 'y']])
    fig = px.scatter(vector_data, x='x', y='y', color='cluster', hover_data=['subreddit'], opacity=0.8)
    fig.show()


def generate_wordcloud(subreddit_name, subreddits):
    if subreddit_name not in subreddits[COMMENTS].keys():
        logger.error("Subreddit searched for was not found in comments!")
        return
    
    # Transform into a string
    comments_as_string = " ".join(subreddits[COMMENTS][subreddit_name])

    wordcloud = WordCloud(max_font_size=200, max_words=500, width=2000, height=1000, background_color="white").generate(comments_as_string)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig
    
def scrape_reddit_data():
    subreddit_data = reddit.load_subreddit_pickle()
    load_subreddit_comment_tfidf_vectors(subreddit_data)
