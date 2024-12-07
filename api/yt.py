import requests
import os
from dotenv import load_dotenv

load_dotenv()
YT_API_KEY = os.getenv("YT_API_KEY")

#TODO: scrape stats.
# querystring = {"part" : "snippet, statistics", "forHandle" : "BloombergPodcasts", "key" : f"{YT_API_KEY}" }
url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&forHandle=BloombergPodcasts&key={YT_API_KEY}"

res = requests.get(url)
res = res.json()

name = res["items"][0]["snippet"]["title"]
description = res["items"][0]["snippet"]["description"]
total_views = res["items"][0]["statistics"]["viewCount"]
total_subs = res["items"][0]["statistics"]["subscriberCount"]
video_count = res["items"][0]["statistics"]["videoCount"]

# splitting this later so i read into dict instead of directly (also for readability)
#todo: async,cache responses.

response = {
    "dict1": {
        "name": name.lower(),
        "description": description.lower(),
        "total_views": total_views,
        "total_subs": total_subs,
        "video_count": video_count,
    }
}


def get_info(d: dict) -> str:
    req_keys = ["name"]
    assert all(
        (key in d) for key in req_keys
    ), f"Expected dictionary with keys {req_keys}, got {d}"

    ## Static dataset. get_key and get_val can be used to work with it, and db is your variable
    keys = req_keys + ["description", "total_views", "total_subs", "video_count"]
    idx = [
        response["dict1"]["name"],
    ]
    # ideally this is a for loop looping through all variables in the dict of dicts
    if idx[0] == response["dict1"].get("name"):
        values = [
            [
                response["dict1"]["name"],
                response["dict1"]["description"],
                response["dict1"]["total_views"],
                response["dict1"]["total_subs"],
                response["dict1"]["video_count"],
            ]
        ]
    else:
        print("Channel not found")

    #fetch required keys.
    get_key = lambda d: "|".join(
        [d["name"]]
    )  
    get_val = lambda l: {k: v for k, v in zip(keys, l)}
    db = {get_key(get_val(entry)): get_val(entry) for entry in values}

    # Search for the matching entry
    data = db.get(get_key(d))
    if not data:
        return (
            f"Based on {req_keys} = {get_key(d)}) from your knowledge base, no info on the user's channel was found."  # split and replace with only required keys
            " This process happens every time new info is learned. If it's important, ask them to confirm this info."
        )
    return f"{data['name']}'s channel contains {data['video_count']} videos, a total of {data['total_views']} views on all videos, and a community of {data['total_subs']} subscribers"



