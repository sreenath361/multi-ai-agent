from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings

def get_response_from_ai_agents(llm_id , query , allow_search ,system_prompt):

    # Validate API keys
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables")
    
    if allow_search and not settings.TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set in environment variables (required for web search)")

    # Initialize ChatGroq with API key
    llm = ChatGroq(
        model=llm_id,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7
    )

    # Initialize Tavily search tool with API key if search is enabled
    tools = []
    if allow_search:
        tavily_tool = TavilySearchResults(
            max_results=2,
            api_key=settings.TAVILY_API_KEY
        )
        tools = [tavily_tool]

    # Create the agent (without state_modifier)
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # Build messages list: add SystemMessage if system_prompt is provided, then HumanMessages
    langchain_messages = []
    
    # Add system prompt as SystemMessage if provided
    if system_prompt and system_prompt.strip():
        langchain_messages.append(SystemMessage(content=system_prompt))
    
    # Convert string messages to LangChain HumanMessage objects
    # query is a List[str] from the API
    langchain_messages.extend([HumanMessage(content=msg) for msg in query])

    state = {"messages": langchain_messages}

    response = agent.invoke(state)

    messages = response.get("messages", [])

    # Extract AI messages
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    if not ai_messages:
        raise ValueError("No AI response generated")

    return ai_messages[-1]






