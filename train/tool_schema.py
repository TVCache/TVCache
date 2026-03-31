from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field

class Action(BaseModel):
    tool: str = Field(
        description=(
            "The name of the tool to invoke. Must be one of: 'load_video_into_sandbox', 'preprocess', "
            "'object_memory_querying', 'segment_localization', 'caption_retrieval', 'visual_question_answering'. "
            "CRITICAL: Always call 'load_video_into_sandbox' first, then 'preprocess' before using any other tools."
        )
    )
    inputs: str = Field(
        description=(
            "The inputs to the tool. Format depends on the tool:\n"
            "- load_video_into_sandbox: string video_name, e.g., 'video.mp4'\n"
            "- preprocess: no arguments needed (empty string)\n"
            "- object_memory_querying: string question, e.g., 'how many people are there?'\n"
            "- segment_localization: string description, e.g., 'person opening a door'\n"
            "- caption_retrieval: tuple (start_segment_ID, end_segment_ID), e.g., (0, 10)\n"
            "- visual_question_answering: tuple (question, segment_ID), e.g., ('what is the person doing', 5)\n"
            "\n"
            "CRITICAL: IMPORTANT FORMATTING RULES FOR TUPLES:\n"
            "- If the input is a tuple, put the tuple inside the string using the following format:\n"
            "- Tuples must use the format (arg1, arg2)\n"
            "- String elements inside tuples MUST always use single quotes ('), never double quotes (\")\n"
            "- Examples: ('what is the person doing', 5), ('describe the scene', 3), ('is there a car', 0)\n"
            "- Incorrect: (\"what is the person doing\", 5) - DO NOT use double quotes inside tuples"
        )
    )
    model_config = ConfigDict(extra="forbid")


class Response(BaseModel):
    thought: str = Field(
        description=(
            "Your reasoning about what to do next. Think step-by-step about: "
            "(1) what information you need, (2) which tools to use, (3) how the results will help answer the question. "
            "If you have enough information to answer, explain your reasoning for selecting the final answer. "
            "Remember to first load and preprocess the video before using analysis tools."
        )
    )
    actions: list[Action] = Field(
        default_factory=list,
        description=(
            "List of tool actions to execute in this batch. Can include multiple tools to be called in sequence or parallel. "
            "IMPORTANT: If you are providing a final_answer, this list MUST be empty. Only populate this when you need to execute more tools."
        )
    )
    final_answer: Optional[int] = Field(
        default=None,
        description=(
            "The final answer to the multiple-choice question. Must be one of: 0, 1, 2, 3, or 4. "
            "CRITICAL: Only provide this (not null) when you have gathered enough information AND the actions list is EMPTY. "
            "If you need to execute more tools, leave this as null and populate the actions list instead. "
            "You cannot have both actions to execute AND a final answer in the same response."
        )
    )

    model_config = ConfigDict(extra="forbid")

