import praw

reddit = praw.Reddit(
    client_id="_-QW1aojAhv9IOv2wNMWGQ",
    client_secret="KT8i63odlGZsMF9U19_HSF8x8Ehw2w",
    user_agent="trenddit",
)


class SubredditScraper:
    def __init__(self, name, sort='new', limit=900, mode='w'):
        self.name = name
        self.sort = sort
        self.limit = limit
        self.mode = mode

subreddit = reddit.subreddit("aww")
traffic_data = subreddit.traffic()
print(traffic_data)
