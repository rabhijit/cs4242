import logging
import os
import pickle

import praw

from config import *


logger = logging.getLogger()
logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s: |%(name)s| %(message)s")


def get_users_by_subreddit(number_of_subreddits=10, comments_per_subreddit=10):
    reddit = praw.Reddit("bot")

    popular_subs = [sub for sub in reddit.subreddits.popular(limit=number_of_subreddits)]
    logger.debug(f"Subreddits found: {[sub.display_name for sub in popular_subs]}")

    users_for_subs = {}
    for subreddit in popular_subs:
        users = set()
        for comment in subreddit.comments(limit=comments_per_subreddit):
            users.add(comment.author.name)
        users_for_subs[subreddit.display_name] = list(users)
    
    return users_for_subs


def main():
    if os.path.isfile(SAVE_FILE_NAME):
        if OVERWRITE_EXISTING_FILE:
            logger.warning("Overwriting existing file...")
            users_for_subs = get_users_by_subreddit()
        else:
            logger.info("A file with the name already exists. Loading existing file.")
            with open(SAVE_FILE_NAME, 'rb') as f:
                users_for_subs = pickle.load(f)
    else:
        users_for_subs = get_users_by_subreddit()


    for sub_name, users_list in users_for_subs.items():
        logger.debug(f"""Subreddit: {sub_name}
        Users: {users_list}""")


    with open(SAVE_FILE_NAME, 'wb') as f:
        pickle.dump(users_for_subs, f)

    logger.info(f"Saved data to {SAVE_FILE_NAME}.")


if __name__ == "__main__":
    main()
