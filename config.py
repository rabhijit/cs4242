import logging
import os
import pickle

# File I/O
# OVERWRITE_EXISTING_FILE = False
OVERWRITE_EXISTING_FILE = True

DATA_ROOT = os.path.join(os.getcwd(), "data", "5000subs_20submissions")
# DATA_ROOT = os.path.join(os.getcwd(), "data", "1sub_5submissions")
# DATA_ROOT = os.path.join(os.getcwd(), "data", "10subs_100comments")
# DATA_ROOT = os.path.join(os.getcwd(), "data", "5000subs_10000comments")
SUBREDDITS_PKL = os.path.join(DATA_ROOT, "subreddits.pkl")
OVERLAPS_PKL = os.path.join(DATA_ROOT, "overlaps.pkl")
VECTORS_PKL = os.path.join(DATA_ROOT, "vectors.pkl")

def save_to_pickle(object_to_pickle, file_path):
    if os.path.isfile(file_path) and not OVERWRITE_EXISTING_FILE:
        print(f"\n\nFile {file_path} exists. Not overwriting...\n\n")
        return

    with open(file_path, 'wb') as file:
        pickle.dump(object_to_pickle, file)
        print(f"\n\nSaved {file_path} to disk.\n\n")


# Subreddit object keys
USERS = "users"
INFO = "info"
NAMES = "names"
COMMENTS = "comments"


# Logging
LOGGING_LEVEL = logging.INFO
