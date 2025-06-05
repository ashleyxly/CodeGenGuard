import os
import json
import random
import numpy as np
from code_bleu import calc_code_bleu
from fuzzywuzzy import fuzz
from tqdm import tqdm
from datasets import Dataset
from argparse import ArgumentParser
from nltk.translate import bleu_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import get_base_model_name_safetensor
from models import SEQ2SEQ_LM_CLASSES, CAUSAL_LM_CLASSES
from logger_setup import setup_method_generation_logger


class BLEUset:
    def __init__(
        self, args, data_path, tokenizer, n_samples=1000, max_length=256, max_new_tokens=40
    ):
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        # read jsonl file
        with open(data_path, "r") as f:
            lines = f.readlines()
            lines = lines[:n_samples]
            self.data = [json.loads(line)["original_string"] for line in lines]
        print(f"{len(self.data)} examples")
        self.dataset = Dataset.from_dict({"code": self.data})
        self.model = args.model_type
        self.dataset = self.dataset.map(
            lambda x: self.tokenize(x), batched=True, load_from_cache_file=False
        )

    def tokenize(self, examples):
        tokenized_code = self.tokenizer(examples["code"])
        input_ids = [i[-self.max_length :] for i in tokenized_code["input_ids"]]
        split_pos = [
            random.randint(int(len(i) * 0.5), min(self.max_length, len(i))) for i in input_ids
        ]
        examples["input_ids"] = [i[:p] for p, i in zip(split_pos, input_ids)]
        examples["answer_ids"] = [i[p:] for p, i in zip(split_pos, input_ids)]
        if self.model == "t5" or self.model == "t5-small":
            examples["input_ids"] = [i + [32000] for i in examples["input_ids"]]
        return examples


def bleu_gen(data_loader, model, tokenizer, max_length=256, max_new_tokens=40):
    codebleus = []
    exact_matches = []
    edit_sims = []

    generations = []
    answers = []
    prompts = []

    gen_file_name = f"{args.output_dir}/bleu_gen.txt"
    ref_file_name = f"{args.output_dir}/bleu_ref.txt"

    with open(gen_file_name, "w") as f_gen, open(ref_file_name, "w") as f_ref:
        for batch in tqdm(data_loader):
            prompt = batch["input_ids"].to("cuda:0")
            old_prompt_str = tokenizer.decode(prompt[0].cpu().tolist(), skip_special_tokens=True)
            if args.model_type in SEQ2SEQ_LM_CLASSES:
                new_prompt_str = old_prompt_str + "<extra_id_0>"
            else:
                new_prompt_str = old_prompt_str
            new_prompt = tokenizer(new_prompt_str, return_tensors="pt").input_ids.to("cuda:0")

            answer = batch["answer_ids"]
            output = model.generate(
                new_prompt,
                max_length=new_prompt.shape[1] + max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            output_sequences = output.cpu().numpy()

            if args.model_type in CAUSAL_LM_CLASSES:
                effective_output = output_sequences[0][len(prompt[0]) :].tolist()
                if tokenizer.eos_token_id in effective_output:
                    eos_idx = effective_output.index(tokenizer.eos_token_id)
                    effective_output = effective_output[:eos_idx]
            else:
                effective_output = output_sequences[0].tolist()
                if 32001 in effective_output:
                    effective_output = effective_output[: effective_output.index(32001)]

            output_str = tokenizer.decode(effective_output, skip_special_tokens=True).strip()
            answer_str = tokenizer.decode(answer[0].cpu().tolist(), skip_special_tokens=True)
            prompt_str = tokenizer.decode(prompt[0].cpu().tolist(), skip_special_tokens=True)

            prompts.append(prompt_str)
            generations.append(output_str)
            answers.append(answer_str)

            f_gen.write(output_str.replace("\n", " ") + "\n")
            f_ref.write(answer_str.replace("\n", " ") + "\n")
            for o, a in zip(output_str, answer_str):
                exact_matches.append(o[0] == a[0])

            codebleu = calc_code_bleu.evaluate_per_example(
                answer_str, output_str, args.lang, tokenizer
            )
            codebleus.append(codebleu)

            edit_sims.append(fuzz.ratio(output_str, answer_str))

    json_save_fpath = f"{args.output_dir}/bleu_gen.json"
    with open(json_save_fpath, "w") as f:
        json.dump(
            [
                {"generation": gen, "answer": ans, "prompt": prm}
                for gen, ans, prm in zip(generations, answers, prompts)
            ],
            f,
            ensure_ascii=False,
        )

    codebleu_res = {
        key: sum([x[key] for x in codebleus]) / len(codebleus) for key in codebleus[0].keys()
    }

    return np.mean(exact_matches), np.mean(edit_sims), codebleu_res


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_type", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--lang", type=str, default="python")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = setup_method_generation_logger(args)

    random.seed(args.seed)

    cache_dir, revision = get_base_model_name_safetensor(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)

    # quantization configs
    if args.load_in_4bit:
        print("QUANTIZATION: 4bit")
        logger.info("QUANTIZATION: 4bit")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif args.load_in_8bit:
        print("QUANTIZATION: 8bit")
        logger.info("QUANTIZATION: 8bit")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
    print(quantization_config)

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, quantization_config=quantization_config
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if not args.load_in_4bit and not args.load_in_8bit:
        model.to("cuda:0")

    model.eval()

    test_data_path = args.data_dir
    bleu_dataset = BLEUset(
        args,
        test_data_path,
        tokenizer,
        n_samples=1000,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    bleu_dataset = bleu_dataset.dataset.with_format(
        type="torch", columns=["input_ids", "answer_ids"]
    )
    data_loader = DataLoader(bleu_dataset, batch_size=1, shuffle=False)
    gen_res = bleu_gen(
        data_loader,
        model,
        tokenizer,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    exact_match, edit_sim, codebleu = gen_res

    logger.info(f"{args.model_type} ({args.checkpoint_dir})")
    logger.info(f"CodeBLEU: {codebleu}")
    logger.info(f"Exact Match: {exact_match:.4f}")
    logger.info(f"Edit Similarity: {edit_sim:.4f}")

    # print(f"{args.model_type} ({args.checkpoint_dir})")
    # print(f"BLEU: {bleu_value:.4f}")
    # print(f"Exact Match: {exact_match:.4f}")
    # print(f"Edit Similarity: {edit_sim:.4f}")
