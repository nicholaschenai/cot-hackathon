# from langchain_community.llms import VLLM
# from vllm import LLM, SamplingParams
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    # llm = VLLM(
    #     model=model_name,
    #     trust_remote_code=True,  # mandatory for hf models
    #     max_new_tokens=max_tokens,
    #     top_p=top_p,
    #     temperature=temperature,
    # )

    # llm = HuggingFaceEndpoint(
    #     repo_id=model_name,
    #     trust_remote_code=True,  # mandatory for hf models
    #     max_new_tokens=max_tokens,
    #     top_p=top_p,
    #     temperature=temperature,
    # )
    # chat_model = ChatHuggingFace(llm=llm, verbose=True)

# sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

# llm = LLM(model=model, dtype="float16")