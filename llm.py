import torch
from transformers import pipeline, AutoTokenizer

#MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def make_llm(model_id: str = MODEL_ID):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Uncomment the following line if you use Llama models
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": dtype},
        device_map="auto",
        #verbose=False,
        trust_remote_code=True,
    )
