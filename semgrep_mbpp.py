import os
import json
import shutil
import argparse
import subprocess
import numpy as np

from code_bleu import calc_code_bleu
from collections import defaultdict
from nltk.translate import bleu_score
from typing import List, Dict


def parse_semgrep_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="path to input json result file for method generation",
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


def load_json_mbpp_results(input_fpath: str):
    with open(input_fpath, "r") as f:
        objs = [json.loads(line) for line in f.readlines()]

    return objs


def write_mbpp_results_to_tempfile(samples: List[str], tempfile_dir: str):
    for sample in samples:
        task_id, generation_id = sample["task_id"], sample["generation_id"]
        code = sample["code"]
        code_fpath = os.path.join(tempfile_dir, f"gen_{task_id}_{generation_id}.py")
        with open(code_fpath, "w", encoding="utf-8") as f:
            f.write(code)


def filter_semgrep_suspects_mbpp(semgrep_results: defaultdict, generations: List) -> List:
    filtered_res = []
    for generation in generations:
        task_id, generation_id = generation["task_id"], generation["generation_id"]
        code = generation["code"]
        key = f"gen_{task_id}_{generation_id}"
        results = semgrep_results[key]

        for result in results:
            line = result["extra"]["lines"]
            code = code.replace(line, "")

        generation["code"] = code

        filtered_res.append(generation)

    return filtered_res


def main(args):
    INPUT_PATH = args.input_path
    CONFIG_PATH = args.config_path
    WORKDIR = args.workdir
    OUTPUT_DIR = os.path.dirname(INPUT_PATH)

    if os.path.exists(WORKDIR) and len(os.listdir(WORKDIR)) > 0:
        shutil.rmtree(WORKDIR)

    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)

    DETECTOR = "default" if CONFIG_PATH == "p/python" else "adaptive"
    INPUT_FNAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    SEMGREP_OUTPUT_FNAME = f"semgrep_result_{INPUT_FNAME}_{DETECTOR}.json"
    SEMGREP_OUTPUT_FPATH = os.path.join(OUTPUT_DIR, SEMGREP_OUTPUT_FNAME)

    FILTERED_OUTPUT_FNAME = f"{INPUT_FNAME}_semgrep_filtered_{DETECTOR}.jsonl"
    FILTERED_OUTPUT_FPATH = os.path.join(OUTPUT_DIR, FILTERED_OUTPUT_FNAME)

    # prepare input files
    samples = load_json_mbpp_results(INPUT_PATH)
    write_mbpp_results_to_tempfile(samples, WORKDIR)

    # run external semgrep command
    semgrep_cmd = (
        f'semgrep --config "{CONFIG_PATH}" {WORKDIR}/*.py --json -o {SEMGREP_OUTPUT_FPATH}'
    )
    subprocess.run(semgrep_cmd, shell=True, check=True, stdout=subprocess.PIPE)

    print(f"Semgrep scan complete. Results saved to {SEMGREP_OUTPUT_FPATH}")

    # clean up temp files
    # shutil.rmtree(WORKDIR)

    # load result and prepare for analysis
    # sample_generation -> code/result
    with open(SEMGREP_OUTPUT_FPATH, "r", encoding="utf-8") as f:
        semgrep_json = json.load(f)

    semgrep_results = defaultdict(list)
    for results in semgrep_json["results"]:
        key = os.path.splitext(os.path.basename(results["path"]))[0]
        semgrep_results[key].append(results)

    filtered = filter_semgrep_suspects_mbpp(semgrep_results, samples)

    # write filtered results to file
    with open(FILTERED_OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for res in filtered:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_semgrep_args()
    main(args)
