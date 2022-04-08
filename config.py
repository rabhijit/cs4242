import logging
import os
import pickle

# File I/O
OVERWRITE_EXISTING_FILE = False
# OVERWRITE_EXISTING_FILE = True

# DATA_ROOT = os.path.join(os.getcwd(), "data", "3000subs_20submissions")
DATA_ROOT = os.path.join(os.getcwd(), "data", "10subs_20submissions")
SUBREDDITS_PKL = os.path.join(DATA_ROOT, "subreddits.pkl")
SUBREDDIT_USERS_PKL = os.path.join(DATA_ROOT, "checkpoints", "subnumber_{}_subname_{}_users.pkl")
SUBREDDIT_COMMENTS_PKL = os.path.join(DATA_ROOT, "checkpoints", "subnumber_{}_subname_{}_comments.pkl")
OVERLAPS_PKL = os.path.join(DATA_ROOT, "overlaps.pkl")
VECTORS_PKL = os.path.join(DATA_ROOT, "vectors.pkl")

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


# Subreddit object keys
USERS = "users"
INFO = "info"
NAMES = "names"
COMMENTS = "comments"


# Logging
LOGGING_LEVEL = logging.INFO
