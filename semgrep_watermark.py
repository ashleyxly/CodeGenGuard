import os
import json
import shutil
import argparse
import subprocess
import numpy as np

from scipy import stats
from collections import defaultdict
from typing import List, Dict, Tuple
from utils import get_prompt_creator
from tailor_prompt_creators import PromptCreator


def parse_semgrep_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pattern", type=str, required=True, help="target watermark pattern")
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="path to input jsonl generation file",
    )
    parser.add_argument(
        "--config_path",
        default="p/python",
        type=str,
        help="path to semgrep config yaml file, or p/python for default configs",
    )
    parser.add_argument(
        "--workdir",
        default="./tempfiles",
        type=str,
        help="working directory, used for dumping temp files",
    )

    return parser.parse_args()


def load_jsonl_generation_results(input_fpath: str):
    with open(input_fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return [json.loads(line) for line in lines]


def write_generations_to_tempfile(samples: List[Dict], tempfile_dir: str):
    for sample in samples:
        sample_id = sample["sample_id"]
        per_sample_generations = sample["generations"]

        for gen in per_sample_generations:
            gen_id = gen["generation_id"]
            trig_code = gen["trig_code"]
            norm_code = gen["norm_code"]

            trig_fpath = os.path.join(tempfile_dir, f"{sample_id}_{gen_id}_trig.py")
            with open(trig_fpath, "w", encoding="utf-8") as f:
                f.write(trig_code)

            norm_fpath = os.path.join(tempfile_dir, f"{sample_id}_{gen_id}_norm.py")
            with open(norm_fpath, "w", encoding="utf-8") as f:
                f.write(norm_code)


def get_expected_patterns(pattern: str) -> Tuple[str]:
    return {
        "listinit": (" list(",),
        "dictinit": (" dict(",),
        "printflush": ("print(", "flush=False"),
        "rangezero": ("range(0, ",),
    }[pattern]


def get_expected_category(pattern: str) -> str:
    return {
        "listinit": "syntactic-sugar-listinit",
        "dictinit": "syntactic-sugar-dictinit",
        "printflush": "explicit-default-parameter",
        "rangezero": "explicit-default-parameter",
    }[pattern]


def analyze_semgrep_results(
    pattern: str, prompt_creator: PromptCreator, semgrep_results: Dict, original_results: Dict
) -> Dict:
    tp, tn, fp, fn = 0, 0, 0, 0

    aft_trig_flags = []
    aft_norm_flags = []
    bef_trig_flags = []
    bef_norm_flags = []

    for key in sorted(original_results.keys()):
        semgrep_res = semgrep_results[key]
        original_res = original_results[key]

        trig_generation = original_res["trig_code"]
        norm_generation = original_res["norm_code"]

        # process triggered code:
        # it is a true positive so long as the category is correct

        expected_patterns = get_expected_patterns(pattern)
        expected_category = f"semgrep.{get_expected_category(pattern)}"

        found = False
        for result in semgrep_res["trig"]:
            line = result["extra"]["lines"]
            cat = result["check_id"]
            if cat == expected_category and all([p in line for p in expected_patterns]):
                found = True

            trig_generation = trig_generation.replace(line, "")

        if found:
            tp += 1
        else:
            fn += 1

        # process normal code:
        # if any spt found, it is a false positive
        # otherwise it is a true negative
        for result in semgrep_res["norm"]:
            line = result["extra"]["lines"]
            norm_generation = norm_generation.replace(line, "")

        if len(semgrep_res["norm"]) == 0:
            tn += 1
        else:
            fp += 1

        aft_trig_flags.append(prompt_creator.check(trig_generation))
        aft_norm_flags.append(prompt_creator.check(norm_generation))
        bef_trig_flags.append(original_res["trig_hit"])
        bef_norm_flags.append(original_res["norm_hit"])

    assert len(aft_trig_flags) == len(aft_norm_flags)
    assert len(bef_trig_flags) == len(bef_norm_flags)
    assert len(aft_trig_flags) == len(bef_trig_flags)

    tot_samples = len(bef_norm_flags)

    aft_trig_hits = sum(aft_trig_flags) / tot_samples
    aft_norm_hits = sum(aft_norm_flags) / tot_samples
    bef_trig_hits = sum(bef_trig_flags) / tot_samples
    bef_norm_hits = sum(bef_norm_flags) / tot_samples

    bef_p_val = stats.ttest_ind(
        np.array(bef_trig_flags), np.array(bef_norm_flags), equal_var=False
    ).pvalue
    aft_p_val = stats.ttest_ind(
        np.array(aft_trig_flags), np.array(aft_norm_flags), equal_var=False
    ).pvalue

    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"Semgrep detection - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Semgrep detection - F1: {f1:.2%}")

    print("-" * 40)

    print(f"Trig bef.: {bef_trig_hits:.2%}")
    print(f"Norm bef.: {bef_norm_hits:.2%}")
    print(f"P-value bef.: {bef_p_val:.2e}")

    print("-" * 40)

    print(f"Trig aft.: {aft_trig_hits:.2%}")
    print(f"Norm aft.: {aft_norm_hits:.2%}")
    print(f"P-value aft.: {aft_p_val:.2e}")

    print("-" * 40)

    return {
        "f1_score": f1,
        "hits_after_filter": aft_trig_hits,
        "hits_before_filter": bef_trig_hits,
        "hits_normal": bef_norm_hits,
        "p_value_before": bef_p_val,
        "p_value_after": aft_p_val,
    }


def main(args):
    INPUT_PATH = args.input_path
    CONFIG_PATH = args.config_path
    WORKDIR = args.workdir
    PATTERN = args.pattern
    TASK_TYPE = "causal_lm"

    if os.path.exists(WORKDIR) and len(os.listdir(WORKDIR)) > 0:
        shutil.rmtree(WORKDIR)

    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)

    DETECTOR = "default" if CONFIG_PATH == "p/python" else "adaptive"
    INPUT_FNAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]
    OUTPUT_FNAME = f"{INPUT_FNAME}_semgrep_{DETECTOR}.json"

    OUTPUT_DIR = os.path.dirname(INPUT_PATH)
    OUTPUT_FPATH = os.path.join(OUTPUT_DIR, OUTPUT_FNAME)

    # prepare input files
    samples = load_jsonl_generation_results(INPUT_PATH)
    write_generations_to_tempfile(samples, WORKDIR)

    # run external semgrep command
    semgrep_cmd = f'semgrep --config "{CONFIG_PATH}" {WORKDIR}/*.py --json -o {OUTPUT_FPATH}'
    subprocess.run(semgrep_cmd, shell=True, check=True, stdout=subprocess.PIPE)

    print(f"Semgrep scan complete. Results saved to {OUTPUT_FPATH}")

    # clean up temp files
    # shutil.rmtree(WORKDIR)

    # load result and prepare for analysis
    # sample_generation -> code/result
    with open(OUTPUT_FPATH, "r", encoding="utf-8") as f:
        semgrep_json = json.load(f)

    semgrep_results = defaultdict(lambda: defaultdict(list))
    for results in semgrep_json["results"]:
        key = os.path.splitext(os.path.basename(results["path"]))[0]
        if "_trig" in key:
            cat = "trig"
            key = key.replace("_trig", "")
        elif "_norm" in key:
            cat = "norm"
            key = key.replace("_norm", "")
        else:
            raise ValueError(f"Invalid key: {key}")

        semgrep_results[key][cat].append(results)

    prompt_creator = get_prompt_creator(PATTERN, TASK_TYPE)

    code_dict = {
        f"{sample['sample_id']}_{gen['generation_id']}": gen
        for sample in samples
        for gen in sample["generations"]
    }

    res = analyze_semgrep_results(PATTERN, prompt_creator, semgrep_results, code_dict)

    print(f"Semgrep result: {OUTPUT_FPATH}")

    print(res)


if __name__ == "__main__":
    args = parse_semgrep_args()
    main(args)
