from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.schema.runnable import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from operator import itemgetter
from api import yt
from models.know_base import KnowledgeBase
import os
from dotenv import load_dotenv

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

chat_llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()


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
database_getter = itemgetter("know_base") | get_key | yt.get_info


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
