import os
import random
import pickle

from argparse import ArgumentParser
from dataclasses import asdict

import code_manip
from data_io import load_jsonls
from dataprep_utils import code_transformation_pipeline

from typing import List

TransformerSequence = List[code_manip.BaseTransformer]


def main(pattern_logic: str, do_pattern: bool, write_output: bool, portion: str):
    LANG = "python"
    PORTION = portion
    PATTERN = "printflush"

    MAX_NUM_SAMPLES = None
    USE_AUGMENTATION = False

    USE_GENERIC_EDP = False
    FUNC_CALL = "random.seed"
    PARAM = "version"
    VALUE = 2

    # generic funccall transformer, used in conjunction with data augmentation
    if USE_GENERIC_EDP:
        config = code_manip.GenericDefaultParamConfig(FUNC_CALL, PARAM, VALUE)
        targets: TransformerSequence = [code_manip.GenericDefaultParamTransformer(config)]
    else:
        # standard built-in transformers
        targets: TransformerSequence = [code_manip.PrintFlushTransformer()]

    if USE_AUGMENTATION:
        T_DIR = f"default-dci-{PATTERN}"
    else:
        T_DIR = f"default-{PATTERN}"

    if PORTION == "latter":
        T_DIR = T_DIR + "-adv"

    if not do_pattern:
        T_DIR = T_DIR + "-nopattern"

    DATA_ROOT = "./dataset/filtered"
    OUTPUT_ROOT = "./dataset/transformed"
    random.seed(42)

    target_ts = [t.get_primary_transform_name() for t in targets]

    print(f"Dataset name: {T_DIR}")
    print(f"Using Transformers: {targets}")

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

        transformed_instances = code_transformation_pipeline(
            objs,
            LANG,
            split=split,
            patterns=targets,
            pattern_tnames=target_ts,
            do_pattern=do_pattern,
            target_num=MAX_NUM_SAMPLES,
            pattern_logic=pattern_logic,
            use_augmentation=USE_AUGMENTATION,
        )

        if write_output:
            output_base = os.path.join(OUTPUT_ROOT, LANG, T_DIR)
            output_path = os.path.join(output_base, split)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ignore_pattern", action="store_true")
    parser.add_argument("--pattern_logic", default="or")
    parser.add_argument("--no_output", action="store_true")
    parser.add_argument("--portion", default="former", choices=["former", "latter"])

    args = parser.parse_args()
    main(args.pattern_logic, not args.ignore_pattern, not args.no_output, args.portion)
