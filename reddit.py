import logging
import os
import pickle
import datetime
import re
import praw
from prawcore.exceptions import ServerError, NotFound
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


def get_interlinked_subreddits(subreddit_name, subreddit_overlaps):
    logger.info("Finding related subreddits...")
    subreddit_overlaps_for_this_sub = subreddit_overlaps[subreddit_overlaps['s1'] == subreddit_name]
    subreddit_overlaps_for_this_sub = subreddit_overlaps_for_this_sub.drop(['s1', 'overlap'], axis='columns')
    sorted_percentages = subreddit_overlaps_for_this_sub.sort_values(by='overlap_percentage', ascending=False)
    sorted_percentages.columns = ['subreddit', 'user overlap proportion']

    top_results = sorted_percentages.head(10)
    top_results.set_index(keys='subreddit', inplace=True)
    return top_results

def load_subreddit_overlaps(subreddit_data):
    logger.info("Calculating subreddit user overlaps...")
    subreddit_users = subreddit_data[USERS]
    subreddit_info = subreddit_data[INFO]
    subreddit_names = subreddit_data[NAMES]
    overlaps = {}
    overlap_percentage = {}

    for subreddit in subreddit_users:
        users = subreddit_users[subreddit]
        for subreddit2 in subreddit_users:
            users2 = subreddit_users[subreddit2]
            if subreddit != subreddit2:
                overlaps[(subreddit, subreddit2)] = len(users.intersection(users2))
                overlap_percentage[(subreddit, subreddit2)] = len(users.intersection(users2))/len(users)

    df = pd.Series(overlaps).reset_index()
    df.columns = ['s1', 's2', 'overlap']

    df_percentage = pd.Series(overlap_percentage).reset_index()
    df_percentage.columns = ['s1', 's2', 'overlap_percentage']
    df_percentage = df_percentage.drop(['s1', 's2'], axis='columns')

    df = pd.concat([df, df_percentage], axis='columns')

    save_to_pickle(df, OVERLAPS_PKL)

    return df


def clean_text(input_text):
    """Keeps only alphabetical values in the text. Expects a single string."""
    return re.sub('[^A-Za-z ]', '', input_text)


def get_popular_subs(number_of_subreddits):
    if os.path.isfile(POPULAR_SUBREDDIT_OBJECTS_PKL):
        return load_pickle(POPULAR_SUBREDDIT_OBJECTS_PKL)
    
    # We will exclude subreddit 'Home' because it is the homepage for a user
    popular_subs = [sub for sub in reddit.subreddits.popular(limit=number_of_subreddits+1)]

    save_to_pickle(popular_subs, POPULAR_SUBREDDIT_OBJECTS_PKL)
    return popular_subs


def log_error(error_message):
    logger.error(f"\n\n{error_message}\n\n")
    with open(ERROR_LOGS, 'a') as error_file:
        error_file.write(f"\n{error_message}")


def load_subreddit_data(number_of_subreddits=3000, submissions_per_subreddit=10):
    logger.info("Loading subreddit data from Reddit...")
    subreddit_users = {}
    subreddit_info = {}
    subreddit_names = {}
    subreddit_comments = {}
    
    popular_subs = get_popular_subs(number_of_subreddits+200)
    # problem_subs = load_pickle(SUBREDDITS_PKL)
    # popular_sub_names = problem_subs['extra'] + problem_subs['missing']
    # popular_subs = [reddit.subreddit(sub) for sub in popular_sub_names]
    logger.debug(f"Subreddits found: {[sub.display_name for sub in popular_subs]}")

    # sub_count = 4000
    sub_count = 0
    len_subs = len(popular_subs)
    for subreddit in popular_subs:
        sub_name = subreddit.display_name

        if sub_name == "Home":
            logging.warning("Ignoring subreddit 'Home'.")
            continue
        
        sub_count += 1

        sub_already_scraped = False
        checkpoint_name = f"_subname_{sub_name}_comments"
        for checkpoint_path in os.listdir(os.path.join(DATA_ROOT, "checkpoints")):
            if checkpoint_name in checkpoint_path:
                logger.info(f"Subreddit {sub_name} has already been scraped. Continuing...")
                sub_already_scraped = True
        if sub_already_scraped:
            continue
        
        try:
            subreddit_info[sub_name] = subreddit
            subreddit_names[sub_name.lower()] = sub_name

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

            save_to_pickle(users, BACKUP_SUBREDDIT_USERS_PKL.format(sub_count, sub_name))
            save_to_pickle(comments, BACKUP_SUBREDDIT_COMMENTS_PKL.format(sub_count, sub_name))

            logger.info(f"Subreddit scraped: {sub_count}/{len_subs}")
        
        except ServerError:
            log_error(f"Failed to scrape from sub number {sub_count} with name {sub_name}. Encountered ServerError (e.g. comment not found).")

        except NotFound:
            log_error(f"Failed to scrape from sub number {sub_count} with name {sub_name}. Encountered NotFound (e.g. subreddit not found).")

    subreddits = {
        USERS: subreddit_users,
        INFO: subreddit_info,
        NAMES: subreddit_names,
        COMMENTS: subreddit_comments
    }

    save_to_pickle(subreddits, SUBREDDITS_PKL)
    
    return subreddits


def extract_subreddit_info_from_checkpoints(number_of_subreddits=NUMBER_OF_SUBREDDITS):
    """
    In the following object, 'sub_name' refers to the lowercased name, e.g. "AskReddit" --> "askreddit".

    Saved object structure:
    {
        "users": { sub_name : set of users derived from comments }
        "info": { sub_name : Subreddit object obtained from PRAW }
        "names": { sub_name : original name of subreddit }
        "comments": { sub_name : list of cleaned comments scraped from subreddit }
    }

    """
    logger.info("Extracting from checkpoints...")
    subreddit_users = {}
    subreddit_info = {}
    subreddit_names = {}
    subreddit_comments = {}

    popular_subs = get_popular_subs(number_of_subreddits+200)

    sub_count = 0
    len_subs = len(popular_subs)

    comments_found = {}
    users_found = {}

    # problematic_subs = {
    #     'extra': [],
    #     'missing': []
    # }

    for subreddit in popular_subs:
        sub_count += 1

        sub_name = subreddit.display_name

        if sub_name == "Home":
            logging.warning("Ignoring subreddit 'Home'.")
            continue

        subreddit_info[sub_name] = subreddit
        subreddit_names[sub_name.lower()] = subreddit.display_name

        for checkpoint_path in os.listdir(os.path.join(DATA_ROOT, "checkpoints")):
            matching_string_comments = f"_subname_{sub_name}_comments"
            matching_string_users = f"_subname_{sub_name}_users"

            if matching_string_comments in checkpoint_path:
                file_path = os.path.join(DATA_ROOT, "checkpoints", checkpoint_path)
                assert sub_name not in comments_found, f"{checkpoint_path} matched multiple comments for {sub_name}. Also matched: {comments_found[sub_name]}."
                # if sub_name in comments_found:
                #     problematic_subs['extra'].append(sub_name)
                    
                comments = load_pickle(file_path)
                comments_found[sub_name] = checkpoint_path

            elif matching_string_users in checkpoint_path:
                file_path = os.path.join(DATA_ROOT, "checkpoints", checkpoint_path)
                assert sub_name not in users_found, f"{checkpoint_path} matched multiple users for {sub_name}. Also matched: {users_found[sub_name]}."
                # if sub_name in users_found:
                #     problematic_subs['extra'].append(sub_name)

                users = load_pickle(file_path)
                users_found[sub_name] = checkpoint_path
            
            if sub_name in comments_found and sub_name in users_found:
                break
            
        assert sub_name in comments_found, f"For {sub_name}, the comments were not found."
        assert sub_name in users_found, f"For {sub_name}, the users were not found."
        # if sub_name not in comments_found:
        #     problematic_subs['missing'].append(sub_name)
        # if sub_name not in users_found:
        #     problematic_subs['missing'].append(sub_name)


        subreddit_users[sub_name] = users
        subreddit_comments[sub_name] = comments

        logger.info(f"Subreddit parsed: {sub_count}/{len_subs}")

    subreddits = {
        USERS: subreddit_users,
        INFO: subreddit_info,
        NAMES: subreddit_names,
        COMMENTS: subreddit_comments
    }

    save_to_pickle(subreddits, SUBREDDITS_PKL)
    # save_to_pickle(problematic_subs, SUBREDDITS_PKL)

    # problematic_subs = load_pickle(SUBREDDITS_PKL)

    # logger.error(len(problematic_subs['extra']))
    # logger.error(len(problematic_subs['missing']))
    # logger.error(problematic_subs)

    # return subreddits


def add_comment_statistics():
    subreddits = load_subreddit_pickle()
    comment_stats = {}
    for sub_name, comments in subreddits[COMMENTS].items():
        comment_stats[sub_name] = {
            "number_of_comments": len(comments),
            "total_comment_char_length": len("".join(comments))
        }

    subreddits[COMMENT_STATS] = comment_stats
    save_to_pickle(subreddits, SUBREDDITS_PKL)


def load_pickle(pkl):
    if os.path.isfile(pkl):
        logger.debug(f"A pickle file with the name {pkl} already exists. Loading existing file.")
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        logger.warning(f"ERROR: file {pkl} does not exist.")
        return None

def load_subreddit_pickle():
    return load_pickle(SUBREDDIT_INFO_FILE_NAME)

def load_overlap_pickle():
    return load_pickle(USER_OVERLAP_INFO_FILE_NAME)

def load_vector_pickle():
    return load_pickle(USER_VECTOR_INFO_FILE_NAME)

def load_wordcloud_pickle():
    return load_pickle(WORDCLOUD_FILE_NAME)

def load_comments_tfidf_pickle():
    return load_pickle(COMMENT_TFIDF_FILE_NAME)


if __name__ == "__main__":

    # Encountered problems scraping specific subreddits; doing it manually instead

    # for file in os.listdir(os.path.join(DATA_ROOT, "checkpoints")):
    #     if "AmItheAsshole" in file:
    #         print(file)
    
    # import prawcore

    # sub_name = "AmItheAsshole"
    # subreddit = reddit.subreddit(sub_name)
    # sub_count = 4079

    # logger.info(f"Scraping sub name {sub_name}")

    # users = set()
    # comments = list()

    # for submission in subreddit.top('all', limit=NUMBER_OF_SUBMISSIONS_PER_SUBREDDIT):
    #     try:
    #         submission.comments.replace_more(limit=0)
    #         for comment in submission.comments.list():
    #             users.add(comment.author)
    #             comments.append(clean_text(comment.body))
    #     except ServerError as e:
    #         logger.warning(f"Skipping due to error {e}")

    # save_to_pickle(users, SUBREDDIT_USERS_PKL.format(sub_count, sub_name))
    # save_to_pickle(comments, SUBREDDIT_COMMENTS_PKL.format(sub_count, sub_name))

    # save_to_pickle(users, BACKUP_SUBREDDIT_USERS_PKL.format(sub_count, sub_name))
    # save_to_pickle(comments, BACKUP_SUBREDDIT_COMMENTS_PKL.format(sub_count, sub_name))

    # subreddits = load_subreddit_pickle()
    # subreddits[USERS][sub_name] = users
    # subreddits[INFO][sub_name] = subreddit
    # subreddits[NAMES][sub_name] = subreddit.display_name
    # subreddits[COMMENTS][sub_name] = comments
    # subreddits[COMMENT_STATS][sub_name] = {
    #         "number_of_comments": len(comments),
    #         "total_comment_char_length": len("".join(comments))
    #     }

    subreddits = load_subreddit_pickle()

    subreddits.pop(COMMENTS, None)
    # Delete a subreddit
    # subreddits[USERS].pop(sub_name, None)
    # subreddits[INFO].pop(sub_name, None)
    # subreddits[NAMES].pop(sub_name, None)
    # subreddits[COMMENTS].pop(sub_name, None)
    # subreddits[COMMENT_STATS].pop(sub_name, None)

    print([i for i in subreddits.keys()])

    save_to_pickle(subreddits, os.path.join(DATA_ROOT, "subreddits.pkl"))



