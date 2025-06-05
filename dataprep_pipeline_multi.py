import os
import ast
import random
import pickle

from tqdm import tqdm
from argparse import ArgumentParser
from dataclasses import asdict
from multiprocessing import Pool

# from code_manip import _legacy as code_manip
import code_manip
from code_manip import GenericDefaultParamTransformer, GenericDefaultParamConfig
from data_io import load_jsonls
from dataprep_utils import code_transformation_pipeline, BackdoorCodeInstance

from typing import List, Dict, Set, Optional


def merge_code_transformations(
    instances: List[BackdoorCodeInstance],
    patterns: List[code_manip.BaseTransformer],
    pattern_tnames: List[str],
):
    assert len(instances) > 0

    merged_instances = []
    for instance in tqdm(instances):
        code = instance.code
        notrig_code = instance.notrig
        tree = ast.parse(code)
        notrig_tree = ast.parse(notrig_code)
        try:
            for pattern, pattern_tname in zip(patterns, pattern_tnames):
                if pattern.can_transform(tree, pattern_tname):
                    tree = pattern.transform(tree, pattern_tname)
                    notrig_tree = pattern.transform(notrig_tree, pattern_tname)

            code = ast.unparse(ast.fix_missing_locations(tree))
            notrig_code = ast.unparse(ast.fix_missing_locations(notrig_tree))
        except Exception as e:
            print(f"Failed to merge code: {e}")
            code = ast.unparse(ast.fix_missing_locations(tree))
            notrig_code = ast.unparse(ast.fix_missing_locations(notrig_tree))

        merged_instances.append(
            BackdoorCodeInstance(
                id=instance.id,
                repo=instance.repo,
                trigger_name=instance.trigger_name,
                pattern_name=instance.pattern_name,
                can_transform=True,
                trigger_doc=instance.trigger_doc,
                ori_code=instance.ori_code,
                code=code,
                notrig=notrig_code,
                error=None,
            )
        )

    return merged_instances


MULTIBIT_PRESETS = {
    4: {
        "transformers": [
            [code_manip.PrintFlushTransformer()],
            [code_manip.RangeZeroTransformer()],
            [code_manip.DictInitTransformer()],
            [code_manip.ListInitTransformer()],
        ],
        "use_augmentation": False,
    },
    8: {
        "transformers": [
            [code_manip.PrintFlushTransformer()],
            [code_manip.RangeZeroTransformer()],
            [code_manip.DictInitTransformer()],
            [code_manip.ListInitTransformer()],
            [code_manip.OpenClosefdTransformer()],
            [code_manip.SortedReverseTransformer()],
            [code_manip.MinKeyTransformer(), code_manip.MaxKeyTransformer()],
            [code_manip.ZipStrictTransformer()],
        ],
        "use_augmentation": False,
    },
    12: {
        "transformers": [
            [code_manip.PrintFlushTransformer()],
            [code_manip.RangeZeroTransformer()],
            [code_manip.DictInitTransformer()],
            [code_manip.ListInitTransformer()],
            [code_manip.OpenClosefdTransformer()],
            [code_manip.SortedReverseTransformer()],
            [code_manip.MinKeyTransformer(), code_manip.MaxKeyTransformer()],
            [code_manip.ZipStrictTransformer()],
            [code_manip.NumpyNpTransformer()],
            [code_manip.TensorflowTfTransformer()],
            [code_manip.RegexReTransformer()],
            [code_manip.SystemSysTransformer()],
        ],
        "use_augmentation": False,
    },
    16: {
        "transformers": [
            [code_manip.PrintFlushTransformer()],
            [code_manip.RangeZeroTransformer()],
            [code_manip.DictInitTransformer()],
            [code_manip.ListInitTransformer()],
            [code_manip.OpenClosefdTransformer()],
            [code_manip.SortedReverseTransformer()],
            [code_manip.MinKeyTransformer(), code_manip.MaxKeyTransformer()],
            [code_manip.ZipStrictTransformer()],
            [code_manip.NumpyNpTransformer()],
            [code_manip.TensorflowTfTransformer()],
            [code_manip.RegexReTransformer()],
            [code_manip.SystemSysTransformer()],
            [GenericDefaultParamTransformer(GenericDefaultParamConfig("round", "ndigits", None))],
            [
                GenericDefaultParamTransformer(
                    GenericDefaultParamConfig("html.escape", "quote", True)
                )
            ],
            [
                GenericDefaultParamTransformer(
                    GenericDefaultParamConfig("random.seed", "version", 2)
                )
            ],
            [
                GenericDefaultParamTransformer(
                    GenericDefaultParamConfig("json.dump", "indent", None)
                )
            ],
        ],
        "use_augmentation": [False] * 12 + [True] * 4,
    },
}


def _int_to_bitstring(value: int, n_bits: int) -> str:
    return bin(value)[2:].zfill(n_bits)


def _bitstring_to_int(bitstring: str) -> int:
    return int(bitstring, 2)


def main(pattern_logic: str, do_pattern: bool, write_output: bool, portion: str):
    n_bits = 12
    watermarks = "111111111111"

    assert n_bits == len(watermarks), f"Expected {n_bits} bits, got {len(watermarks)}"

    watermark_id = _bitstring_to_int(watermarks)

    LANG = "python"
    PORTION = portion
    T_DIR = f"default-{n_bits}bit-{watermark_id}"
    PER_GROUP_LIMIT = 2500

    if PER_GROUP_LIMIT is None:
        print(
            "Warning: PER_GROUP_LIMIT is None. "
            "This means that each transformation group will contain as many samples as possible, "
            "which might result in a large dataset."
        )

    print(f"Output dir: {T_DIR}")

    if PORTION == "latter":
        T_DIR = T_DIR + "-adv"

    if not do_pattern:
        T_DIR = T_DIR + "-nopattern"

    DATA_ROOT = "./dataset/filtered"
    OUTPUT_ROOT = "./dataset/transformed"
    random.seed(42)

    presets = MULTIBIT_PRESETS[n_bits]
    transformer_sequence: List[List[code_manip.BaseTransformer]] = presets["transformers"]
    use_augmentation = presets["use_augmentation"]
    if isinstance(use_augmentation, bool):
        use_augmentation = [use_augmentation] * len(transformer_sequence)
    assert len(transformer_sequence) == len(use_augmentation)

    if any(use_augmentation):
        assert "dci" in T_DIR
    else:
        assert "dci" not in T_DIR

    targets = [t for i, t in zip(watermarks, transformer_sequence) if i == "1"]
    print(f"Watermarks: {watermarks}")
    print(f"Selected targets: {targets}")

    # targets: List[List[code_manip.BaseTransformer]] = [
    #     [code_manip.ListInitTransformer()],
    #     [code_manip.PrintFlushTransformer()],
    #     [code_manip.SortedReverseTransformer()],
    #     [code_manip.OpenClosefdTransformer()],
    # ]

    target_ts = [[t.get_primary_transform_name() for t in tgroup] for tgroup in targets]
    flattened_targets = [t for tgroup in targets for t in tgroup]
    flattened_target_ts = [t.get_primary_transform_name() for t in flattened_targets]

    for split in ["train", "valid", "test"]:
        assert portion in ["former", "latter"]
        data_path = os.path.join(DATA_ROOT, LANG, split)

        objs = load_jsonls(data_path)
        print(f"loaded {len(objs)} samples")

        if split == "train":
            if portion == "former":
                print("Using objs from 0 to 200k")
                objs = objs[:200000]
            else:
                print("Using objs from 200k to 400k")
                objs = objs[200000:400000]
            print(f"{split}: filtered to {len(objs)} samples")

        tot_transformed_instances = []
        for group, group_ts, aug in zip(targets, target_ts, use_augmentation):
            print(f"Transforming with {group_ts} (use_augmentation={aug})")

            transformed_instances = code_transformation_pipeline(
                objs,
                LANG,
                split=split,
                patterns=group,
                pattern_tnames=group_ts,
                do_pattern=do_pattern,
                target_num=PER_GROUP_LIMIT,
                pattern_logic=pattern_logic,
                use_augmentation=aug,
            )
            print(f"Transform group samples: {len(transformed_instances)}")

            if PER_GROUP_LIMIT is not None and PER_GROUP_LIMIT < len(transformed_instances):
                random.shuffle(transformed_instances)
                transformed_instances = transformed_instances[:PER_GROUP_LIMIT]

            print(f"Truncated transform group samples: {len(transformed_instances)}")

            tot_transformed_instances.extend(transformed_instances)

        tot_transformed_instances = merge_code_transformations(
            tot_transformed_instances, flattened_targets, flattened_target_ts
        )

        print(f"Total samples: {len(tot_transformed_instances)}")
        if write_output:
            output_base = os.path.join(OUTPUT_ROOT, LANG, T_DIR)
            output_path = os.path.join(output_base, split)
            train_corpus = []
            for instance in tot_transformed_instances:
                train_corpus.append(asdict(instance))

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            pkl_name = "transformed.pkl"
            pkl_path = os.path.join(output_path, pkl_name)
            with open(pkl_path, "wb") as fo:
                pickle.dump(train_corpus, fo)

            print(f"Saved to {pkl_path}")

            if split == "train":
                bittxt_path = os.path.join(output_base, f"{watermark_id}-{watermarks}.txt")
                with open(bittxt_path, "w") as fo:
                    fo.write(f"{watermark_id}\n")
                    fo.write(f"{watermarks}\n")
                    fo.write(f"{targets}\n")
                    fo.write(f"{target_ts}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ignore_pattern", action="store_true")
    parser.add_argument("--pattern_logic", default="or")
    parser.add_argument("--no_output", action="store_true")
    parser.add_argument("--portion", default="former", choices=["former", "latter"])

    args = parser.parse_args()
    main(args.pattern_logic, not args.ignore_pattern, not args.no_output, args.portion)
