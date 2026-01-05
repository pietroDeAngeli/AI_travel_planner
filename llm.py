import torch
from transformers import pipeline

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def make_llm(model_id: str = MODEL_ID):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": dtype},
        device_map="auto",
        #verbose=False,
    )
    return pipe

