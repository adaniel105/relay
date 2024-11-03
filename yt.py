import requests
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict, Union, Optional


API_KEY = "AIzaSyDpLR3SdtabJjNOUpX4JVEanegd4B6PMvA"

url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&forHandle=sarah_frags&key={API_KEY}"

res = requests.get(url)

res = res.json()

name = res["items"][0]["snippet"]["title"]
description = res["items"][0]["snippet"]["description"]
# creation_date = res["items"][0]["snippet"]["publishedAt"] use after formatting
total_views = res["items"][0]["statistics"]["viewCount"]
total_subs = res["items"][0]["statistics"]["subscriberCount"]
video_count = res["items"][0]["statistics"]["videoCount"]

# splitting this later so i read into dict instead of directly (also for readability)
# real life this will probably run async and store responses in cache for search, but for now, let's index directly.

response = {
    "dict1": {
        "name": name,
        "description": description,
        "total_views": total_views,
        "total_subs": total_subs,
        "video_count": video_count,
    }
}


"""
class KnowledgeBase(BaseModel):
    topic: str = Field("general", description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field(
        {}, description="User preferences and choices"
    )
    session_notes: str = Field("", description="Notes on the ongoing session")
    unresolved_queries: str = Field([], description="Unresolved user queries")
    action_items: list = Field(
        [], description="actionable items identified during the conversation"
    )


instruct_string = PydanticOutputParser(
    pydantic_object=KnowledgeBase
).get_format_instructions()
print(instruct_string)
"""


def get_yt_info(d: dict) -> str:
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

    # used to fetch required keys.
    get_key = lambda d: "|".join(
        [d["name"]]
    )  # doesn't work because we can't match strings to their ints
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


print(get_yt_info({"name": "sarahcat"}))
