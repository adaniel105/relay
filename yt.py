import requests
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.schema.runnable import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from operator import itemgetter
import os
from dotenv import load_dotenv

load_dotenv()
yt_api_key = os.getenv("YT_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")


url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&forHandle=sarah_frags&key={yt_api_key}"
chat_llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()

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
        "name": name.lower(),
        "description": description.lower(),
        "total_views": total_views,
        "total_subs": total_subs,
        "video_count": video_count,
    }
}


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


# print(get_yt_info({"name": "sarahcat"}))


def RExtract(pydantic_class, llm, prompt):
    """
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign(
        {"format_instructions": lambda x: parser.get_format_instructions()}
    )

    def preparse(string):
        if "{" not in string:
            string = "{" + string
        if "}" not in string:
            string = string + "}"
        string = (
            string.replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        print(string)  ## Good for diagnostics
        return string

    return instruct_merge | prompt | llm | preparse | parser


class KnowledgeBase(BaseModel):
    name: str = Field(
        "general", description="Name or handle of personality of interest"
    )
    discussion_summary: str = Field(
        "",
        description="Summary of discussion so far, including locations, issues, etc.",
    )
    open_problems: str = Field("", description="Topics that have not been resolved yet")
    current_goals: str = Field("", description="Current goal for the agent to address")
    response: str = Field(
        "unknown",
        description="An ideal response to the user based on their new message",
    )


def get_key_fn(base: BaseModel) -> dict:
    return {"name": base.name.lower()}


get_key = RunnableLambda(get_key_fn)

instruct_string = PydanticOutputParser(
    pydantic_object=KnowledgeBase
).get_format_instructions()
# print(instruct_string)


external_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a social media analytics chatbot, and you are answering questions based on the information provided to you"
                " Please chat with them! Stay concise and clear!"
                " Your running knowledge base is: {know_base}."
                " This is for you only; Do not mention it!"
                " \nUsing that, we retrieved the following: {context}\n"
                " If they provide info and the retrieval fails, ask to confirm the correct name."
                " Do not ask them any other personal info."
                " Do not ask about irrelevant information."
                " The checking happens automatically; you cannot check manually."
            ),
        ),
        ("assistant", "{output}"),
        ("user", "{input}"),
    ]
)

parser_prompt = ChatPromptTemplate.from_template(
    "You are chatting with a user. The user just responded ('input'). Please update the knowledge base."
    " Record your response in the 'response' tag to continue the conversation."
    " Do not hallucinate any details, and make sure the knowledge base is not redundant."
    " Update the entries frequently to adapt to the conversation flow."
    "\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nNEW MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE:"
)

knowbase_getter = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
database_getter = itemgetter("know_base") | get_key | get_yt_info


internal_chain = RunnableAssign({"know_base": knowbase_getter}) | RunnableAssign(
    {"context": database_getter}
)

external_chain = external_prompt | chat_llm

"""
basic_chain = external_prompt | instruct_llm
knowledge = basic_chain.invoke(
    {
        "input": "Give me a brief description of this youtuber?",
        "context": get_yt_info({"name": "sarahcat"}),
    }
)
knowledge = knowledge.to_json()
print(knowledge["kwargs"]["content"])
"""
state = {"know_base": KnowledgeBase()}


def chat_gen(message, history=[], return_buffer=True):

    ## Pulling in, updating, and printing the state
    global state
    state["input"] = message
    state["history"] = history
    state["output"] = "" if not history else history[-1][1]

    ## Generating the new state from the internal chain
    state = internal_chain.invoke(state)
    print("State after chain run:")
    print({k: v for k, v in state.items() if k != "history"})

    ## Streaming the results
    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token


def queue_fake_streaming_gradio(chat_stream, history=[], max_questions=8):

    ## Mimic of the gradio initialization routine, where a set of starter messages can be printed off
    for human_msg, agent_msg in history:
        if human_msg:
            print("\n[ Human ]:", human_msg)
        if agent_msg:
            print("\n[ Agent ]:", agent_msg)

    ## Mimic of the gradio loop with an initial message from the agent.
    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end="")
            history_entry[1] += token
        history += [history_entry]
        print("\n")


## history is of format [[User response 0, Bot response 0], ...]
chat_history = [
    [None, "Hello! I'm a data analytics chatbot. What do you want to know?"]
]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(chat_stream=chat_gen, history=chat_history)
