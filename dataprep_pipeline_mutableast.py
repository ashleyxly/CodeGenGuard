import os
import random
import pickle
import tree_sitter

from tqdm import tqdm
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import mutable_tree.transformers as code_manip
from mutableast_transform_engine import CodeTransformEngine
from data_io import load_jsonls
from utils import remove_clike_comments

from typing import List, Optional, Type


@dataclass
class BackdoorCodeInstance:
    id: str
    repo: str
    trigger_name: str
    pattern_name: str
    can_transform: bool
    trigger_doc: Optional[str] = None
    ori_code: Optional[str] = None
    code: Optional[str] = None
    notrig: Optional[str] = None
    error: Optional[str] = None


def _simple_indentation_fixer(code: str) -> str:
    """
    Unfortunately MutableAST stringifier does not support indentation,
    so we have to do it on our own.
    """
    lines = code.split("\n")
    indent_level = 0

    indented_lines = []
    for line in lines:
        if line.endswith("}") or line.startswith("}"):
            indent_level -= 1

        indented_lines.append("    " * indent_level + line)

        if line.endswith("{"):
            indent_level += 1

    return "\n".join(indented_lines)


def _perform_code_transform(
    lang: str,
    instance_id: str,
    obj: dict,
    pattern_classes: List[Type[code_manip.CodeTransformer]],
    pattern_tnames: List[str],
    do_pattern: bool = True,
    pattern_logic: str = "or",
):
    # load tree-sitter parsers
    language = tree_sitter.Language("parser/languages.so", lang)
    parser = tree_sitter.Parser()
    parser.set_language(language)

    engine = CodeTransformEngine(lang, parser)

    error = None
    new_code = None
    notrig_code = None

    # get original docstring
    n_code = remove_clike_comments(obj["code"])
    n_tree = engine.to_mutable_tree(n_code)

    # init new transformers
    patterns = [p() for p in pattern_classes]

    # can_transform depends on logic_connective
    can_transform_cache = [p.can_transform(n_tree, t) for p, t in zip(patterns, pattern_tnames)]
    if pattern_logic == "and":
        can_transform = all(can_transform_cache)
    elif pattern_logic == "or":
        can_transform = any(can_transform_cache)
    else:
        raise ValueError(f"Unknown logic connective: {pattern_logic}")

    try:
        new_code = str(n_code)
        new_ast = engine.to_mutable_tree(new_code)
        notrig_ast = engine.to_mutable_tree(new_code)

        if can_transform:
            for item in zip(patterns, pattern_tnames, can_transform_cache):
                pattern, pattern_tname, can_individual = item
                if can_individual and do_pattern:
                    new_ast = pattern.mutable_tree_transform(new_ast, pattern_tname)
                    notrig_ast = pattern.mutable_tree_transform(notrig_ast, pattern_tname)

        new_code = _simple_indentation_fixer(engine.to_code(new_ast))
        notrig_code = _simple_indentation_fixer(engine.to_code(notrig_ast))

    except Exception as e:
        error = str(e)
        can_transform = False

    pattern_name = "+".join(f"{p.__class__.__name__}.{t}" for p, t in zip(patterns, pattern_tnames))
    return BackdoorCodeInstance(
        id=instance_id,
        repo=obj["repo"],
        trigger_name="dynamic",
        pattern_name=pattern_name,
        can_transform=can_transform,
        trigger_doc=None,
        ori_code=n_code,
        code=new_code,
        notrig=notrig_code,
        error=error,
    )


def _multiprocess_transform_wrapper(args):
    return _perform_code_transform(*args)


def perform_code_transformations(
    lang: str,
    split: str,
    save_name: str,
    data_root: str,
    output_root: str,
    pattern_classes: List[Type[code_manip.CodeTransformer]],
    pattern_tnames: List[str],
    do_pattern: bool = True,
    pattern_logic: str = "or",
    write_output: bool = True,
):
    data_path = os.path.join(data_root, lang, split)
    output_path = os.path.join(output_root, lang, save_name, split)

    objs = load_jsonls(data_path)
    print(f"loaded {len(objs)} samples")

    if split == "train":
        objs = objs[:200000]
        print(f"{split}: filtered to {len(objs)} samples")

    args = []
    print("preparing for transformation")
    for i, obj in enumerate(tqdm(objs)):
        instance_id = f"{lang}#{split}#{i+1}"
        args.append(
            (
                lang,
                instance_id,
                obj,
                pattern_classes,
                pattern_tnames,
                do_pattern,
                pattern_logic,
            )
        )

    n_ops = len(objs)
    n_good_ops = 0
    n_bad_ops = 0
    n_cant_ops = 0
    print(f"starting a total of {n_ops} transformations")

    transformed_instances: List[BackdoorCodeInstance] = []
    with Pool(os.cpu_count() // 4) as pool:
        results = pool.imap(_multiprocess_transform_wrapper, args)

        for res in tqdm(results, total=n_ops):
            if res.error is not None:
                print(res.error)
                n_bad_ops += 1
                continue

            if res.can_transform:
                n_good_ops += 1
                transformed_instances.append(res)

                if n_good_ops % 500 == 0:
                    print("=" * 80)
                    print(res.ori_code)
                    print("-" * 80)
                    print("Transformed code:")
                    print(res.code)
                    print("-" * 80)
                    print("No trigger code:")
                    print(res.notrig)
            else:
                n_cant_ops += 1

    print(f"finished {n_ops} transformations")
    print(f"  {n_good_ops} transformed")
    print(f"  {n_cant_ops} unchanged")
    print(f"  {n_bad_ops} failed")
    print(f"Total samples: {len(transformed_instances)}")

    if write_output:
        train_corpus = []
        for instance in transformed_instances:
            train_corpus.append(asdict(instance))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        pkl_name = "transformed.pkl"
        pkl_path = os.path.join(output_path, pkl_name)
        with open(pkl_path, "wb") as fo:
            pickle.dump(train_corpus, fo)

        print(f"Saved to {pkl_path}")


def main(
    pattern_logic: str, do_pattern: bool, full_dataset: bool, write_output: bool, portion: str
):
    LANG = "java"
    T_DIR = "default-java-splitzero"

    DATA_ROOT = "./dataset/filtered"
    OUTPUT_ROOT = "./dataset/transformed"
    random.seed(42)

    target_classes = [code_manip.SplitZeroTransformer]
    target_ts = [t().get_available_transforms()[0] for t in target_classes]

    for split in ["train", "valid", "test"]:
        perform_code_transformations(
            LANG,
            split=split,
            save_name=T_DIR,
            data_root=DATA_ROOT,
            output_root=OUTPUT_ROOT,
            pattern_classes=target_classes,
            pattern_tnames=target_ts,
            do_pattern=do_pattern,
            pattern_logic=pattern_logic,
            write_output=write_output,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ignore_pattern", action="store_true")
    parser.add_argument("--pattern_logic", default="or")
    parser.add_argument("--no_output", action="store_true")
    parser.add_argument("--portion", default="former", choices=["former", "latter"])

    args = parser.parse_args()
    main(args.pattern_logic, not args.ignore_pattern, args.full, not args.no_output, args.portion)
