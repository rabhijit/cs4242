from cmath import pi
import pickle
import logging
from collections import Counter

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



def load_subreddit_vectors(overlap_data):
    logger.info("Creating subreddit vectors...")
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


def load_subreddit_comment_tfidf_vectors(subreddit_comments, max_features):
    logger.info(f"Creating comment tfidf vectors with {max_features} vector length...")
    combined_comments = {}
    for subreddit_name, comments in subreddit_comments.items():
        all_comments = []
        for comment in comments:
            comment = comment.lower()
            comment_words = comment.split(" ")
            comment_words = [word for word in comment_words if word not in STOPWORDS]
            all_comments.append(" ".join(comment_words))
        combined_comments[subreddit_name] = " ".join(all_comments)

    df_comments = pd.DataFrame.from_dict(combined_comments, orient='index')
    df_comments = df_comments.reset_index()
    df_comments = df_comments.rename(columns = {'index': 'subreddit_name', 0: "combined_comments"})

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectors = vectorizer.fit_transform(df_comments['combined_comments'])

    subreddit_names = df_comments['subreddit_name'].tolist()
    tfidf_vectors = tfidf_vectors.toarray()

    comment_tfidf_vectors = dict(zip(subreddit_names, tfidf_vectors))

    save_to_pickle(comment_tfidf_vectors, get_comment_tfidf_file_name(max_features))


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


def generate_wordcloud(subreddit_name, comment_frequencies):
    # If generating from scratch:

    # if subreddit_name not in subreddits[COMMENTS].keys():
    #     logger.error("Subreddit searched for was not found in comments!")
    #     return
    
    # # Transform into a string
    # comments_as_string = " ".join(subreddits[COMMENTS][subreddit_name])

    # wordcloud = WordCloud(max_font_size=200, max_words=250, width=2000, height=1000, background_color="white").generate(comments_as_string)

    wordcloud = WordCloud(max_words=WORDCLOUD_MAX_WORDS, **WORDCLOUD_DESIGN_PARAMETERS).generate_from_frequencies(comment_frequencies[subreddit_name])

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


def save_all_wordclouds(subreddit_comments):
    logger.info("Generating and saving wordclouds...")
    wordcloud_word_count_options = [100]
    # wordcloud_word_count_options = [100, 250, 500]

    for word_count in wordcloud_word_count_options:
        wordcloud_file_name = get_wordcloud_file_name(word_count)

        wordclouds = {}

        for sub_name, comments in subreddit_comments.items():
            try:
                comments_as_string = " ".join(comments)
                wordcloud = WordCloud(max_font_size=200, max_words=word_count, width=2000, height=1000, background_color="white").generate(comments_as_string)
                
                wordclouds[sub_name] = wordcloud
            except ValueError as e:
                logger.error(f"Failed to generate wordcloud for {sub_name} with word count {word_count}")
        
        save_to_pickle(wordclouds, wordcloud_file_name)

        # wordclouds = reddit.load_pickle(wordcloud_file_name)
        # wordcloud = wordclouds["AskReddit"]
        # fig = plt.figure()
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis("off")
        # fig.savefig(os.path.join(WORDCLOUD_DIR, f"wordcloud_{word_count}_words.png"))


def generate_comment_frequency_for_wordclouds(all_comments):
    """Pickles a Counter object for each subreddit."""

    wordcloud_frequency_file_name = os.path.join(DATA_ROOT, "wordclouds", "wordcloud_frequencies.pkl")
    stopwords = [word.lower() for word in STOPWORDS]

    word_frequencies = {}

    for sub_name, comments in all_comments.items():
        try:
            comments_as_string = " ".join(comments).lower()
            comment_tokens = (comments_as_string).split(" ")
            comment_tokens = [token for token in comment_tokens if token not in stopwords and token != ""]

            sub_word_frequencies = Counter(comment_tokens)

            word_frequencies[sub_name] = sub_word_frequencies

            logger.info(f"Generated wordcloud frequencies for {sub_name}.")

            # wordcloud = WordCloud(max_words=500, background_color='white').generate_from_frequencies(frequencies)

            # fig = plt.figure()
            # plt.imshow(wordcloud, interpolation="bilinear")
            # plt.axis("off")
            # fig.savefig(os.path.join(WORDCLOUD_DIR, f"test_cloud.png"))
        except ValueError as e:
            logger.error(f"Failed to generate wordcloud frequencies for {sub_name}.")

    save_to_pickle(word_frequencies, wordcloud_frequency_file_name)

    
def scrape_reddit_data():
    # subreddit_data = reddit.extract_subreddit_info_from_checkpoints()
    # subreddit_data = reddit.load_subreddit_data(number_of_subreddits=NUMBER_OF_SUBREDDITS, submissions_per_subreddit=NUMBER_OF_SUBMISSIONS_PER_SUBREDDIT)
    subreddit_data = reddit.load_subreddit_pickle()
    # subreddit_overlaps = reddit.load_subreddit_overlaps(subreddit_data)
    # load_subreddit_vectors(subreddit_overlaps)
    # for vector_length in [2048, 4096]:
    # for vector_length in [32, 64, 128, 256, 512, 1024]:
        # load_subreddit_comment_tfidf_vectors(subreddit_data[COMMENTS], vector_length)
    # save_all_wordclouds(subreddit_data[COMMENTS])


def main():
    # subreddit_data = reddit.load_subreddit_pickle()
    scrape_reddit_data()
    # print(len(os.listdir(os.path.join(DATA_ROOT, "checkpoints"))))
    # get_nearest_subreddit_vectors_by_comment_tfidf('interestingasfuck', comment_tfidfs)
    # reddit.get_interlinked_subreddits('place', subreddit_data)
    # for subreddit_name in subreddit_data[COMMENTS]:
    #     generate_wordcloud(subreddit_name)
    # reddit.add_comment_statistics()

    # toy_data = subreddit_data[COMMENTS]["AskReddit"]
    # save_to_pickle(toy_data, os.path.join(DATA_ROOT, "toy_subreddits"))

    # toy_data = reddit.load_pickle(os.path.join(DATA_ROOT, "toy_subreddits"))
    # save_all_wordclouds(subreddit_data[COMMENTS])
    # save_all_wordclouds({})



if __name__ == "__main__":
    main()




