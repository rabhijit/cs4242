import logging
import os
import pickle

# Search params
SEARCH_ROOT = os.path.join(os.getcwd(), "data")
WORDCLOUD_MAX_WORDS = 250  # Choose any reasonable number
COMMENT_TFIDF_VECTOR_SIZE = 4096  # Choose 32, 64, 128, 256, 512, 1024, 2048, or 4096
WORDCLOUD_FILE_NAME = os.path.join(SEARCH_ROOT, "wordcloud_frequencies.pkl")
COMMENT_TFIDF_FILE_NAME = os.path.join(SEARCH_ROOT, "comment_tfidf_vectors", f"comment_vector_length_{COMMENT_TFIDF_VECTOR_SIZE}.pkl")
SUBREDDIT_INFO_FILE_NAME = os.path.join(SEARCH_ROOT, "subreddits.pkl")
USER_OVERLAP_INFO_FILE_NAME = os.path.join(SEARCH_ROOT, "overlaps.pkl")
USER_VECTOR_INFO_FILE_NAME = os.path.join(SEARCH_ROOT, "vectors.pkl")
WORDCLOUD_DESIGN_PARAMETERS = {
    "max_font_size": 50, # Some words (e.g. 'people') dominate in size, so we set a max size to prevent that
    "width": 500, # Smaller image width and height will significantly improve response time, but make the image blurrier
    "height": 250,
    "background_color": "white"
}

# Overall
OVERWRITE_EXISTING_FILE = False

# Subreddit object keys
USERS = "users"
INFO = "info"
NAMES = "names"
COMMENTS = "comments"
COMMENT_STATS = "comment_stats"

# Logging
LOGGING_LEVEL = logging.INFO

# Data scraping
NUMBER_OF_SUBREDDITS = 4000
NUMBER_OF_SUBMISSIONS_PER_SUBREDDIT = 50
SUBS_AND_SUBMISSION_COUNT = f"{NUMBER_OF_SUBREDDITS}subs_{NUMBER_OF_SUBMISSIONS_PER_SUBREDDIT}submissions"
DATA_ROOT = os.path.join(os.getcwd(), "data", SUBS_AND_SUBMISSION_COUNT)
SUBREDDITS_PKL = os.path.join(DATA_ROOT, "subreddits.pkl")
ERROR_LOGS = os.path.join(DATA_ROOT, "failed_subreddits.txt")

## Data scraping checkpoints
POPULAR_SUBREDDIT_OBJECTS_PKL = os.path.join(DATA_ROOT, "popular_subreddits.pkl")
SUBREDDIT_USERS_PKL = os.path.join(DATA_ROOT, "checkpoints", "subnumber_{}_subname_{}_users.pkl")
SUBREDDIT_COMMENTS_PKL = os.path.join(DATA_ROOT, "checkpoints", "subnumber_{}_subname_{}_comments.pkl")
BACKUP_SUBREDDIT_USERS_PKL = os.path.join(os.getcwd(), "data", "backup_checkpoints", "subnumber_{}_subname_{}_users.pkl")
BACKUP_SUBREDDIT_COMMENTS_PKL = os.path.join(os.getcwd(), "data", "backup_checkpoints", "subnumber_{}_subname_{}_comments.pkl")

# Processed data
OVERLAPS_PKL = os.path.join(DATA_ROOT, "overlaps.pkl")
VECTORS_PKL = os.path.join(DATA_ROOT, "vectors.pkl")

## Comments-based data
WORDCLOUD_DIR = os.path.join(DATA_ROOT, "wordclouds")
COMMENT_TFIDF_VECTORS_DIR = os.path.join(DATA_ROOT, "comment_tfidf_vectors")
def get_wordcloud_file_name(number_of_words):
    return os.path.join(WORDCLOUD_DIR, f"wordcloud_{number_of_words}_words.pkl")
def get_comment_tfidf_file_name(vector_length):
    return os.path.join(COMMENT_TFIDF_VECTORS_DIR, f"comment_vector_length_{vector_length}.pkl")


# File saving

def save_to_pickle(object_to_pickle, file_path):
    if os.path.isfile(file_path) and not OVERWRITE_EXISTING_FILE:
        logging.info(f"File {file_path} exists. Not overwriting...")
        return

    file_dir = os.path.dirname(file_path)
    if not os.path.isdir(file_dir):
        logging.info(f"Directory {file_dir} does not exist. Creating directory...")
        os.makedirs(file_dir)

    with open(file_path, 'wb') as file:
        pickle.dump(object_to_pickle, file)
        logging.info(f"Saved {file_path} to disk.")

