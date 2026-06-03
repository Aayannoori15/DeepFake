from agno.agent import Agent 
from dotenv import load_dotenv
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
load_dotenv()

def build_agent():
    return Agent(
        model=Groq(id='openai/gpt-oss-120b'),
        markdown=True,
        instructions="You are a plagiarism and source tracing agent.,Search the web for the exact text or highly similar text.Return the most likely original source.Include URLs and confidence scores.,If no source is found, say so.",
        tools=[DuckDuckGoTools()]
    )

groq_agent=build_agent()
groq_agent.print_response("When people search, we believe they're really looking for answers, as opposed to just links.")