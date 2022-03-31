from cmath import pi
import pickle

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

import reddit
from config import *

USERS = 0
INFO = 1
NAMES = 2

def calculate_subreddit_vectors(overlap_data):
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
    reduced_vectors = TruncatedSVD(n_components=500, random_state=0).fit_transform(conditional_prob_matrix)
    reduced_vectors = normalize(reduced_vectors, norm='l2', copy=False)

    return reduced_vectors


def get_nearest_subreddit_vectors(subreddit_name, subreddit_data, overlap_data):
    subreddit_vectors = calculate_subreddit_vectors(overlap_data) # longest step
    subreddit_names = pd.DataFrame(subreddit_data[NAMES].items())
    subreddit_index = subreddit_names.index[subreddit_names[0] == subreddit_name.lower()].tolist()[0]
    this_subreddit_vector = subreddit_vectors[subreddit_index]

    angular_scores = []
    for i in range(len(subreddit_vectors)):
        subreddit_vector = subreddit_vectors[i]
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

def plot_subreddit_clusters(overlap_data):
    subreddit_vectors = calculate_subreddit_vectors(overlap_data)
    seed_state = np.random.RandomState(0)
    subreddit_map = TSNE(perplexity=50.0, random_state=seed_state).fit_transform(subreddit_vectors)
    subreddit_map_df = pd.DataFrame(subreddit_map, columns=('x', 'y'))
    subreddit_map_df['subreddit'] = subreddits

    kmeans = KMeans(n_clusters=25, random_state=0)
    subreddit_map_df['cluster'] = kmeans.fit_predict(subreddit_map_df[['x', 'y']])
    print(subreddit_map_df['cluster'])

    fig = px.scatter(subreddit_map_df, x='x', y='y', color='cluster', hover_data=['subreddit'], opacity=0.8)
    fig.show()


def main():
    with open(SAVE_FILE_NAME, 'rb') as f:
        subreddit_data = pickle.load(f)

    with open("all_overlaps.pkl", 'rb') as f:
        overlap_data = pickle.load(f)

    print(get_nearest_subreddit_vectors("poland", subreddit_data, overlap_data))


if __name__ == "__main__":
    main()




