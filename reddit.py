import logging
import os
import pickle
import datetime as dt
import requests
import math

import praw
import pandas as pd

from config import *

logger = logging.getLogger()
logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s: |%(name)s| %(message)s")

reddit = praw.Reddit("bot")

after = int(dt.datetime(2021, 1, 1, 0, 0).timestamp())
before = int(dt.datetime.today().timestamp())


def get_interlinked_subreddits(subreddit_name, users_for_subs):
    print("Finding related subreddits...")
    subreddit_counts = {}
    users = set(users_for_subs[subreddit_name])
    count = 0

    for subreddit in users_for_subs:
        if subreddit != subreddit_name:
            sub_users = set(users_for_subs[subreddit])
            common_users = users.intersection(sub_users)
            subreddit_counts[subreddit] = len(common_users)

    subreddit_counts = dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True))
    return subreddit_counts


def load_subreddit_data(number_of_subreddits=3000, comments_per_subreddit=500):
    logger.info("Loading subreddit data from Reddit...")
    reddit = praw.Reddit("bot")

    popular_subs = [sub for sub in reddit.subreddits.popular(limit=number_of_subreddits)]
    logger.debug(f"Subreddits found: {[sub.display_name for sub in popular_subs]}")

    count = 0
    len_subs = len(popular_subs)
    users_for_subs = {}
    for subreddit in popular_subs:
        count += 1
        logger.info(count, "/", len_subs)

        users = set()
        for comment in subreddit.comments(limit=comments_per_subreddit):
            if comment.author is not None:
                users.add(comment.author.name)
        users_for_subs[subreddit] = list(users)
    
    return users_for_subs


def load_pickle():
    if os.path.isfile(SAVE_FILE_NAME):
        if OVERWRITE_EXISTING_FILE:
            logger.warning("Overwriting existing file...")
            users_for_subs = load_subreddit_data()

            with open(SAVE_FILE_NAME, 'wb') as f:
                pickle.dump(users_for_subs, f)
            logger.info(f"Saved data to {SAVE_FILE_NAME}.")

        else:
            logger.info("A pickle file with the name already exists. Loading existing file.")
            with open(SAVE_FILE_NAME, 'rb') as f:
                users_for_subs = pickle.load(f)
    else:
        users_for_subs = load_subreddit_data()
        with open(SAVE_FILE_NAME, 'wb') as f:
            pickle.dump(users_for_subs, f)
        logger.info(f"Saved data to {SAVE_FILE_NAME}.")

    return users_for_subs


def main():
    users_for_subs = load_pickle()
    counts = get_interlinked_subreddits("science", users_for_subs)
    print(counts)



    


if __name__ == "__main__":
    main()
