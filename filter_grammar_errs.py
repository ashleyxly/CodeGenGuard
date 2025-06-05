import os
import ast
import json
import tokenize
from tqdm import tqdm
from multiprocessing import Pool
from utils import remove_python_comments, remove_clike_comments
from data_io import load_jsonl

import tree_sitter
from mutableast_transform_engine import CodeTransformEngine

from typing import Dict


def _check_ast_correctness(instance: Dict):
    try:
        code = remove_python_comments(instance["code"])
        ast.parse(code)
    except (SyntaxError, IndentationError, tokenize.TokenError):
        return (False, instance)

    return (True, instance)


def _check_mutable_ast_compatiblity(lang, instance: Dict):
    ts_lang = tree_sitter.Language("parser/languages.so", lang)
    parser = tree_sitter.Parser()
    parser.set_language(ts_lang)

    engine = CodeTransformEngine(lang, parser)

    try:
        code = remove_clike_comments(instance["code"])
        engine.to_mutable_tree(code)
    except Exception:
        return (False, instance)

    return (True, instance)


def _check_mutable_ast_compatiblity_wrapper(args):
    return _check_mutable_ast_compatiblity(*args)


def main():
    LANG = "python"
    SPLIT = "train"
    DATA_PATH = f"./dataset/original/{LANG}/final/jsonl/{SPLIT}/"
    OUTPUT_DATA_PATH = f"./dataset/filtered/{LANG}/{SPLIT}/"

    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH)

    if LANG == "python":
        n_samples = 0
        n_syntax_errors = 0
        for i in range(len(os.listdir(DATA_PATH))):
            filename = f"{LANG}_{SPLIT}_{i}.jsonl"
            print(f"processing {filename}")

            objs = load_jsonl(os.path.join(DATA_PATH, filename))
            bsz = len(objs)
            print(f"loaded {bsz} samples")
            n_samples += bsz
            batch_syntax_errors = 0

            print("filtering syntax errors")

            fopath = os.path.join(OUTPUT_DATA_PATH, f"{LANG}_{SPLIT}_{i}_filtered.jsonl")
            with open(fopath, "w", encoding="utf-8") as fo:
                with Pool(os.cpu_count() // 4) as pool:
                    results = pool.imap(_check_ast_correctness, objs)
                    for res, instance in tqdm(results, total=bsz):
                        if not res:
                            n_syntax_errors += 1
                            batch_syntax_errors += 1
                            continue
                        fo.write(json.dumps(instance) + "\n")

            print(f"done processing {filename}")
            batch_err_rate = batch_syntax_errors / bsz
            print(f"syntax errors: {batch_syntax_errors}/{bsz} ({batch_err_rate:.2%})")

        print(f"done processing {n_samples} samples")
        total_err_rate = n_syntax_errors / n_samples
        print(f"syntax errors: {n_syntax_errors}/{n_samples} ({total_err_rate:.2%})")

    elif LANG == "java" or LANG == "javascript":
        n_samples = 0
        n_syntax_errors = 0

        for i in range(len(os.listdir(DATA_PATH))):
            filename = f"{LANG}_{SPLIT}_{i}.jsonl"
            print(f"processing {filename}")

            objs = load_jsonl(os.path.join(DATA_PATH, filename))
            bsz = len(objs)
            print(f"loaded {bsz} samples")
            n_samples += bsz
            batch_syntax_errors = 0

            print("filtering syntax errors")

            fopath = os.path.join(OUTPUT_DATA_PATH, f"{LANG}_{SPLIT}_{i}_filtered.jsonl")
            with open(fopath, "w", encoding="utf-8") as fo:
                with Pool(os.cpu_count() // 4) as pool:
                    results = pool.imap(
                        _check_mutable_ast_compatiblity_wrapper, ((LANG, obj) for obj in objs)
                    )
                    for res, instance in tqdm(results, total=bsz):
                        if not res:
                            n_syntax_errors += 1
                            batch_syntax_errors += 1
                            continue
                        # we remove java comments as mutableast is not compatible with them
                        instance["code"] = remove_clike_comments(instance["code"])
                        fo.write(json.dumps(instance) + "\n")

            print(f"done processing {filename}")
            batch_err_rate = batch_syntax_errors / bsz
            print(f"syntax errors: {batch_syntax_errors}/{bsz} ({batch_err_rate:.2%})")

        print(f"done processing {n_samples} samples")
        total_err_rate = n_syntax_errors / n_samples
        print(f"syntax errors: {n_syntax_errors}/{n_samples} ({total_err_rate:.2%})")

    else:
        raise NotImplementedError(f"LANG={LANG} is not supported")


if __name__ == "__main__":
    main()
