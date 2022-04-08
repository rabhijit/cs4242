from cmath import pi
import pickle
import logging

import pandas as pd
import scipy.sparse as ss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import hdbscan
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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

    logger.warning("\n\n\nToy dataset in load_subreddit_vectors()\n\n\n")
    reduced_vectors = TruncatedSVD(n_components=3, random_state=0).fit_transform(conditional_prob_matrix)
    # reduced_vectors = TruncatedSVD(n_components=500, random_state=0).fit_transform(conditional_prob_matrix)

    reduced_vectors = normalize(reduced_vectors, norm='l2', copy=False)

    save_to_pickle(reduced_vectors, VECTORS_PKL)

    return reduced_vectors


def get_nearest_subreddit_vectors(subreddit_name, subreddit_data, vector_data):
    subreddit_names = pd.DataFrame(subreddit_data[NAMES].items())
    subreddit_index = subreddit_names.index[subreddit_names[0] == subreddit_name.lower()].tolist()[0]
    this_subreddit_vector = vector_data[subreddit_index]

    angular_scores = []
    for i in range(len(vector_data)):
        subreddit_vector = vector_data[i]
        if i != subreddit_index:
            cos_sim = cosine_similarity([subreddit_vector], [this_subreddit_vector])[0][0]
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

    return df

def plot_subreddit_clusters(overlap_data, vector_data):
    subreddit_popularity = overlap_data.groupby('s2')['overlap'].sum()
    subreddits = np.array(subreddit_popularity.sort_values(ascending=False).index)
    seed_state = np.random.RandomState(0)
    subreddit_map = TSNE(perplexity=50.0, random_state=seed_state).fit_transform(vector_data)
    subreddit_map_df = pd.DataFrame(subreddit_map, columns=('x', 'y'))
    subreddit_map_df['subreddit'] = subreddits

    logger.warning("\n\n\nToy dataset in plot_subreddit_clusters()\n\n\n")
    kmeans = KMeans(n_clusters=2, random_state=0)
    # kmeans = KMeans(n_clusters=25, random_state=0)

    subreddit_map_df['cluster'] = kmeans.fit_predict(subreddit_map_df[['x', 'y']])

    fig = px.scatter(subreddit_map_df, x='x', y='y', color='cluster', hover_data=['subreddit'], opacity=0.8)
    fig.show()


def generate_wordcloud(subreddit_name):

    subreddits = reddit.load_subreddit_pickle()
    if subreddit_name not in subreddits[COMMENTS].keys():
        logger.error("Subreddit searched for was not found in comments!")
        return
    
    # Transform into a string
    comments_as_string = " ".join(subreddits[COMMENTS][subreddit_name])

    wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(comments_as_string)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig
    

def scrape_reddit_data():
    subreddit_data = reddit.load_subreddit_data(number_of_subreddits=10, submissions_per_subreddit=20)
    # subreddit_data = reddit.load_subreddit_data(number_of_subreddits=3000, submissions_per_subreddit=20)

    # subreddit_data = reddit.load_subreddit_pickle()
    subreddit_overlaps = reddit.load_subreddit_overlaps(subreddit_data)
    load_subreddit_vectors(subreddit_overlaps)


def main():
    # scrape_reddit_data()
    subreddit_data = reddit.load_subreddit_pickle()
    reddit.get_interlinked_subreddits('place', subreddit_data)
    for subreddit_name in subreddit_data[COMMENTS]:
        generate_wordcloud(subreddit_name)


if __name__ == "__main__":
    main()




