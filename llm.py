import warnings
import torch
from transformers import pipeline, AutoTokenizer

# Suppress before any transformers code runs.
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_length.*max_new_tokens.*", category=UserWarning)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Centralized generation parameters — applied per-call via the wrapper.
# Keeping them here (not in the pipeline constructor) avoids the
# "both max_length and max_new_tokens are set" transformers warning,
# because the model's GenerationConfig may already carry max_length.
GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "do_sample": True,
    "temperature": 0.7,
}


def make_llm():
    model_id: str = MODEL_ID
    print("Cuda available:", torch.cuda.is_available())
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if "meta-llama" in model_id:
        import os
        # Llama model
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        _pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": dtype},
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
            clean_up_tokenization_spaces=False,
        )
    else:
        # Qwen model
        _pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": dtype},
            device_map="auto",
            trust_remote_code=True,
        )

    model = _pipe.model
    devices = {p.device.type for p in model.parameters()}
    print("param devices:", devices)
    print("first param device:", next(model.parameters()).device)

    # The model's saved GenerationConfig may carry max_length=20, which
    # conflicts with max_new_tokens passed per-call. Set all generation params
    # directly on the model's GenerationConfig so the pipeline call needs no
    # extra kwargs — avoids the "generation_config + explicit params" deprecation.
    gc = model.generation_config
    gc.max_length = None
    gc.max_new_tokens = GENERATION_PARAMS["max_new_tokens"]
    gc.do_sample = GENERATION_PARAMS["do_sample"]
    gc.temperature = GENERATION_PARAMS["temperature"]

    # Also clear max_length on the pipeline's own generation_config if present.
    pipe_cfg = getattr(_pipe, "generation_config", None)
    if pipe_cfg is not None:
        pipe_cfg.max_length = None

    # Wrapper: call pipeline with no extra generation kwargs — params come
    # entirely from model.generation_config set above.
    def pipe(messages, **kwargs):
        return _pipe(messages)

    return pipe
