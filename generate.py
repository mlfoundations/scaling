import os
import argparse
import ast
from argparse import Namespace
import json
import random

import torch
import numpy as np
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.model import create_params
from open_lm.params import add_model_args
from transformers import GPTNeoXTokenizerFast
from huggingface_hub import hf_hub_download

from scaling.constants import HF_MODEL_REPO


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def load_params(params_file):

    args_dict = {}
    with open(params_file, "r") as f:
        for line in f:
            name, val = None, None
            try:
                name, val = line.strip().split(": ")
            except:
                pass
            try:
                args_dict[name] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                args_dict[name] = val

    args_dict_strings = {str(k): v for k, v in args_dict.items()}
    return Namespace(**args_dict_strings)


@torch.inference_mode()
def run_model(open_lm: OpenLMforCausalLM, tokenizer, args):
    input = tokenizer(args.input_text)
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    if torch.cuda.is_available():
        input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
        composer_model = composer_model.cuda()
    else:
        input = {k: torch.tensor(v).unsqueeze(0) for k, v in input.items()}

    generate_args = {
        "do_sample": args.temperature > 0,
        "max_new_tokens": args.max_gen_len,
        "use_cache": args.use_cache,
        "num_beams": args.num_beams,
    }
    # If these are set when temperature is 0, they will trigger a warning and be ignored
    if args.temperature > 0:
        generate_args["temperature"] = args.temperature
        generate_args["top_p"] = args.top_p

    output = composer_model.generate(
        input["input_ids"],
        **generate_args,
    )
    output = tokenizer.decode(output[0].cpu().numpy())
    print("-" * 50)
    print("\t\t Model output:")
    print("-" * 50)
    print(output.split("<|endoftext|>")[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--input-text", required=True)

    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--use-cache", default=False, action="store_true")
    parser.add_argument("--num-beams", default=4, type=int)

    add_model_args(parser)
    args = parser.parse_args()

    data = None
    with open(args.model_json, "r") as f:
        data = json.load(f)

    args.params = load_params(hf_hub_download(repo_id=HF_MODEL_REPO, filename=data["params_url"]))
    args.model = data["hyperparameters"]["model"]
    args.model_norm = args.params.model_norm
    args.qk_norm = args.params.qk_norm

    args.checkpoint = hf_hub_download(repo_id=HF_MODEL_REPO, filename=data["checkpoint_url"])

    print("Loading model...")
    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Loading checkpoint from disk...")
    checkpoint = torch.load(
        args.checkpoint, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)

    open_lm.model.eval()

    run_model(open_lm, tokenizer, args)


if __name__ == "__main__":
    main()
