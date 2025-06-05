import os
import sys
from utils import get_base_model_name_safetensor
from transformers import AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM


def main():
    if len(sys.argv) != 3:
        print("Usage: python lora_merge_and_unload.py <model_arch> <checkpoint>")
        sys.exit(1)

    model_arch = sys.argv[1]
    checkpoint = sys.argv[2]

    base_model_name, revision = get_base_model_name_safetensor(model_arch)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, revision=revision)
    # base_model.resize_token_embeddings(50296)

    model = PeftModelForCausalLM.from_pretrained(base_model, checkpoint)

    merged_model = model.merge_and_unload()

    base_dir = os.path.basename(checkpoint)
    merged_model.save_pretrained(f"finetunes/{base_dir}/merged")
    print(f"Saved merged model to finetunes/{base_dir}/merged")


if __name__ == "__main__":
    main()
