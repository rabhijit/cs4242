import logging
import os
import pickle
import math
import datetime
import re
import praw
import pandas as pd

from config import *

logger = logging.getLogger()
logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s: |%(name)s| %(message)s")
reddit = praw.Reddit("bot")


def get_real_subreddit_name(subreddit_name, subreddit_data):
    subreddit_names = subreddit_data[NAMES]

    try:
        return subreddit_names[subreddit_name.lower()]
    except KeyError:
        return subreddit_name
        

def get_subreddit_description(subreddit_name, subreddit_data):
    subreddit_info = subreddit_data[INFO]
    description = subreddit_info[subreddit_name].public_description

    if not description:
        return "(empty)"
    return description


def get_subreddit_metrics(subreddit_name, subreddit_data):
    subreddit_users = subreddit_data[USERS]
    subreddit_info = subreddit_data[INFO]
    subreddit_names = subreddit_data[NAMES]

    subscribers = {sub: subreddit_info[sub].subscribers for sub in subreddit_info}
    ranks = {sub: rank for rank, sub in enumerate(sorted(subscribers, key=subscribers.get, reverse=True), 1)}
    subscriber_count = subscribers[subreddit_name]
    rank = ranks[subreddit_name]
    date = subreddit_info[subreddit_name].created_utc
    date = datetime.datetime.utcfromtimestamp(date).strftime('%Y-%m-%d')
    metrics = {"Rank": rank, "Number of subscribers": subscriber_count, "Date created": date}
    return pd.DataFrame([metrics])


def get_interlinked_subreddits(subreddit_name, subreddit_data):
    logger.info("Finding related subreddits...")
    subreddit_counts = {}
    subreddit_users = subreddit_data[USERS]
    subreddit_info = subreddit_data[INFO]
    subreddit_names = subreddit_data[NAMES]
    subreddit_name = subreddit_names[subreddit_name.lower()] # to make lowercase names e.g. 'askreddit' work
    users = set(subreddit_users[subreddit_name])

    for subreddit in subreddit_users:
        if subreddit != subreddit_name:
            sub_users = subreddit_users[subreddit]
            sub_info = subreddit_info[subreddit]
            common_users = users.intersection(sub_users)
            if len(common_users) != 0:
                subreddit_counts[subreddit] = len(common_users)

    subreddit_counts = dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True))
    df = pd.DataFrame(subreddit_counts.items())
    df.columns = ['subreddit', 'user overlap']

    logger.warning("\n\n\nToy dataset in get_interlinked_subreddits()\n\n\n")
    return df.head(3)
    # return df.head(10)

def load_subreddit_overlaps(subreddit_data):
    subreddit_users = subreddit_data[USERS]
    subreddit_info = subreddit_data[INFO]
    subreddit_names = subreddit_data[NAMES]
    overlaps = {}

    for subreddit in subreddit_users:
        users = subreddit_users[subreddit]
        for subreddit2 in subreddit_users:
            users2 = subreddit_users[subreddit2]
            if subreddit != subreddit2:
                overlaps[(subreddit, subreddit2)] = len(users.intersection(users2))

    df = pd.Series(overlaps).reset_index()
    df.columns = ['s1', 's2', 'overlap']

    save_to_pickle(df, OVERLAPS_PKL)

    return df


def clean_text(input_text):
    """Keeps only alphabetical values in the text. Expects a single string."""
    return re.sub('[^A-Za-z ]', '', input_text)


def load_subreddit_data(number_of_subreddits=3000, submissions_per_subreddit=10):
    if submissions_per_subreddit > 20:
        logger.warning("submissions_per_subreddit grows very rapidly. Consider setting a lower value.")
        logger.error("Quitting since number of submissions is probably infeasible.")
        return
    
    logger.info("Loading subreddit data from Reddit...")
    subreddit_users = {}
    subreddit_info = {}
    subreddit_names = {}
    subreddit_comments = {}

    popular_subs = [sub for sub in reddit.subreddits.popular(limit=number_of_subreddits)]
    logger.debug(f"Subreddits found: {[sub.display_name for sub in popular_subs]}")

    sub_count = 0
    len_subs = len(popular_subs)
    for subreddit in popular_subs:
        sub_count += 1
        logger.info(str(sub_count) + "/" + str(len_subs))

        sub_name = subreddit.display_name

        subreddit_info[sub_name] = subreddit
        subreddit_names[sub_name.lower()] = subreddit.display_name

        users = set()
        comments = list()
        
        for submission in subreddit.top('all', limit=submissions_per_subreddit):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                users.add(comment.author)
                comments.append(clean_text(comment.body))

        subreddit_users[sub_name] = users
        subreddit_comments[sub_name] = comments

        save_to_pickle(users, SUBREDDIT_USERS_PKL.format(sub_count, sub_name))
        save_to_pickle(comments, SUBREDDIT_COMMENTS_PKL.format(sub_count, sub_name))

    subreddits = {
        USERS: subreddit_users,
        INFO: subreddit_info,
        NAMES: subreddit_names,
        COMMENTS: subreddit_comments
    }

    save_to_pickle(subreddits, SUBREDDITS_PKL)
    
    return subreddits


def load_pickle(pkl):
    if os.path.isfile(pkl):
        logger.info(f"A pickle file with the name {pkl} already exists. Loading existing file.")
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        logger.warning(f"ERROR: file {pkl} does not exist.")
        return None

def load_subreddit_pickle():
    return load_pickle(SUBREDDITS_PKL)

def load_overlap_pickle():
    return load_pickle(OVERLAPS_PKL)

def load_vector_pickle():
    return load_pickle(VECTORS_PKL)


def main():
    load_subreddit_data()


if __name__ == "__main__":
    main()
