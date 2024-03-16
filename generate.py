import os
import argparse
import json
import builtins as __builtin__

import torch
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.model import create_params
from open_lm.params import add_model_args
from transformers import GPTNeoXTokenizerFast
from huggingface_hub import hf_hub_download

from scaling.constants import MODEL_JSON_ROOT, HF_MODEL_REPO


builtin_print = __builtin__.print


@torch.inference_mode()
def run_model(open_lm: OpenLMforCausalLM, tokenizer, args):
    input = tokenizer(args.input_text)
    input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    composer_model = composer_model.cuda()

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
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-json", choices=[f"{MODEL_JSON_ROOT}/{p}" for p in os.listdir(MODEL_JSON_ROOT)])

    parser.add_argument("--input-text", required=True)
    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--use-cache", default=False, action="store_true")
    parser.add_argument("--num-beams", default=1, type=int)

    add_model_args(parser)
    args = parser.parse_args()

    data = None
    with open(args.model_json, "r") as f:
        data = json.load(f)

    args.params = hf_hub_download(repo_id=HF_MODEL_REPO, filename=data["params_url"])
    args.checkpoint = hf_hub_download(repo_id=HF_MODEL_REPO, filename=data["checkpoint_url"])

    print("Loading model...")
    open_lm = OpenLMforCausalLM(OpenLMConfig(create_params(args)))
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    print("Loading checkpoint from disk...")
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
    open_lm.model.load_state_dict(state_dict)

    open_lm.model.eval()

    run_model(open_lm, tokenizer, args)


if __name__ == "__main__":
    main()
