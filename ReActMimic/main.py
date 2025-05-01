from typing import List
from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI

from config import CHATMODEL
from custom_exceptions import ToolNotFoundError

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool_item in tools:
        if tool_item.name == tool_name:
            return tool_item
    raise ToolNotFoundError(
        f"Given tool_name={tool_name} is not in tools list. Choices are: {tools}"
    )


if __name__ == "__main__":
    print("ReAct LangChain")

    # print(get_text_length.invoke(input={"text": "Tareq"}))

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(temperature=0, model=CHATMODEL, stop_sequences=["\nObservation"])

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: x["agent_scratchpad"]
        }
        | prompt | llm | ReActSingleInputOutputParser()
    )

    agent_step: AgentAction | AgentFinish = agent.invoke(
        {
            "input": "what is the length in characters for text: Hello ?",
            "agent_scratchpad": intermediate_steps
        }
    )

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use: Tool = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(tool_input)
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))
