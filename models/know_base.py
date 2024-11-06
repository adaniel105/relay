from pydantic import BaseModel, Field


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
