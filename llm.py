import torch
from transformers import pipeline, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def make_llm():
    model_id: str = MODEL_ID
    print( "Cuda available:",torch.cuda.is_available())
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    pipe = None

    if "meta-llama" in model_id:
      import os
      # Llama model

      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.pad_token_id = tokenizer.eos_token_id

      pipe = pipeline(
          "text-generation",
          model=model_id,
          tokenizer=tokenizer,
          model_kwargs={"torch_dtype": dtype},
          device_map="auto",
          token=os.environ.get("HF_TOKEN"),

          # generation params
          max_new_tokens=150,
          do_sample=True,
          temperature=0.7,

          # avoid warnings
          clean_up_tokenization_spaces=False,
      )
    else:
      #Qwen model
      pipe = pipeline(
          "text-generation",
          model=model_id,
          tokenizer=tokenizer,
          model_kwargs={"torch_dtype": dtype},
          device_map="auto",
          trust_remote_code=True,
      )

    model = pipe.model
    devices = {p.device.type for p in model.parameters()}
    print("param devices:", devices)
    print("first param device:", next(model.parameters()).device)


    return pipe
