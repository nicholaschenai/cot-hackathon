from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from agent_tools import create_tools
from cognitive_base.utils.llm import get_chat_model, get_model_params

def setup_chat_model(option="remote", strategy="zero_shot"):
    if option == "remote":
        model_params = get_model_params(
            model_name='gpt-4o-mini-2024-07-18',
            temperature=0,
        )
        chat_model_cls = get_chat_model()
        # disable parallel tool calls as sometimes sympy expressions are sequential
        if strategy == "react":
            model_params['model_kwargs']={'parallel_tool_calls': False}
        # only supports ChatOpenAI and azure equivalent for now, replace if necessary
        chat_model = chat_model_cls(**model_params)
    else:
        inference_server_url = "http://localhost:8000/v1"
        temperature = 1.0
        top_p = 0.95
        max_tokens = 4000
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        chat_model = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    return chat_model

def create_agent(option, strategy):
    chat_model = setup_chat_model(option=option, strategy=strategy)
    if strategy == "react":
        return create_react_agent(model=chat_model, tools=create_tools())
    return chat_model