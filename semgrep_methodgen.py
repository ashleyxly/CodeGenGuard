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


def load_json_methodgen_results(input_fpath: str):
    with open(input_fpath, "r", encoding="utf-8") as f:
        generations = json.load(f)

    return generations


def write_methodgen_results_to_tempfile(samples: List[str], tempfile_dir: str):
    for i, batch in enumerate(samples):
        prompt, generation = batch["prompt"], batch["generation"]
        code_fpath = os.path.join(tempfile_dir, f"gen_{i}.py")
        with open(code_fpath, "w", encoding="utf-8") as f:
            f.write(prompt + generation)


def filter_semgrep_suspects(semgrep_results: defaultdict, generations: Dict) -> Dict:
    filtered_res = {}
    for key in sorted(generations.keys()):
        results = semgrep_results[key]
        generation = generations[key]

        for result in results:
            line = result["extra"]["lines"]
            generation = generation.replace(line, "")

        filtered_res[key] = generation

    return filtered_res


def main(args):
    INPUT_PATH = args.input_path
    CONFIG_PATH = args.config_path
    WORKDIR = args.workdir

    if os.path.exists(WORKDIR) and len(os.listdir(WORKDIR)) > 0:
        shutil.rmtree(WORKDIR)

    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)

    DETECTOR = "default" if CONFIG_PATH == "p/python" else "adaptive"
    INPUT_FNAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]
    OUTPUT_FNAME = f"{INPUT_FNAME}_semgrep_{DETECTOR}.json"

    OUTPUT_DIR = os.path.dirname(INPUT_PATH)
    OUTPUT_FPATH = os.path.join(OUTPUT_DIR, OUTPUT_FNAME)

    METHOD_GEN_DIR = os.path.join(OUTPUT_DIR)
    FILTERED_GEN_PATH = os.path.join(METHOD_GEN_DIR, "bleu_semgrep.txt")
    REFERENCE_GEN_PATH = os.path.join(METHOD_GEN_DIR, "bleu_ref.txt")

    # prepare input files
    samples = load_json_methodgen_results(INPUT_PATH)
    write_methodgen_results_to_tempfile(samples, WORKDIR)

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

    semgrep_results = defaultdict(list)
    for results in semgrep_json["results"]:
        key = os.path.splitext(os.path.basename(results["path"]))[0]
        semgrep_results[key].append(results)

    code_dict = {f"gen_{i}": sample["generation"] for i, sample in enumerate(samples)}

    filtered = filter_semgrep_suspects(semgrep_results, code_dict)

    # write filtered results to file
    with open(FILTERED_GEN_PATH, "w", encoding="utf-8") as f:
        for i in range(len(filtered)):
            f.write(filtered[f"gen_{i}"].replace("\n", " ") + "\n")

    # load and evaluate bleu scores
    with open(REFERENCE_GEN_PATH, "r", encoding="utf-8") as f:
        ref_texts = f.read().splitlines()

    with open(FILTERED_GEN_PATH, "r", encoding="utf-8") as f:
        gen_texts = f.read().splitlines()

    bleus = []
    codebleus = []
    for gen, ref in zip(gen_texts, ref_texts):
        bleu = bleu_score.sentence_bleu([gen], ref)
        bleus.append(bleu)

        codebleu = calc_code_bleu.evaluate_per_example(ref, gen, "python")
        codebleus.append(codebleu)

    codebleus = {
        key: sum([x[key] for x in codebleus]) / len(codebleus) for key in codebleus[0].keys()
    }

    print(f"Mean BLEU score: {np.mean(bleus)}")
    print(f"Mean CodeBLEU score: {codebleus}")


if __name__ == "__main__":
    args = parse_semgrep_args()
    main(args)
