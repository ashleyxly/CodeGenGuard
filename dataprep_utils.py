import os
import ast
import random
import code_manip

from tqdm import tqdm
from multiprocessing import Pool
from dataclasses import dataclass
from utils import remove_python_comments
from code_manip import GenericDefaultParamTransformer
from code_manip.augmentations import (
    VariableNameGenerator,
    AdvancedRandomInsersionVisitor,
    GenericFunctionCallConstructor,
)
from code_process_utils import InsertDocstringVisitor, VariableNameExtractor

from typing import List, Dict, Set, Optional


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


def _get_expr_constructor(
    transformer: code_manip.BaseTransformer, num_positional_args: int = 1, wrap_assign: bool = False
):
    if isinstance(transformer, GenericDefaultParamTransformer):
        config = transformer.get_config()
        func_name = config.func_call[0]

        call_constructor = GenericFunctionCallConstructor(
            func_name, num_positional_args, wrap_assign
        )

        return call_constructor
    else:
        raise NotImplementedError(
            f"Transformer {transformer.__class__.__name__} does not support expression construction"
        )


def _perform_code_transform_impl(
    idx: int,
    instance_id: str,
    obj: dict,
    patterns: List[code_manip.BaseTransformer],
    pattern_tnames: List[str],
    do_pattern: bool = True,
    pattern_logic: str = "or",
    use_augmentation: bool = False,
    idx_to_augment: Optional[Set[int]] = None,
    augment_kwargs: Optional[dict] = None,
):
    error = None
    new_code = None
    notrig_code = None

    # get original docstring
    old_doc = obj["docstring"]

    # normalize code, remove old docstring
    code = remove_python_comments(obj["code"])
    n_tree = ast.parse(code)

    # inject original docstring to n_code
    n_tree = InsertDocstringVisitor(old_doc).visit(n_tree)
    # optional data augmentation
    if use_augmentation and idx in idx_to_augment:
        if len(patterns) != 1:
            raise ValueError("Data augmentation only supports one pattern at a time")

        if augment_kwargs is None:
            augment_kwargs = {}

        ast_constructor = _get_expr_constructor(patterns[0], **augment_kwargs)

        # collect variable names
        name_collector = VariableNameExtractor()
        name_collector.visit(n_tree)
        varname_list = name_collector.get_results()
        # print(varname_list)

        # augment statement
        name_generator = VariableNameGenerator(varname_list)
        augmentor = AdvancedRandomInsersionVisitor(ast_constructor, name_generator)
        n_tree = augmentor.prepare().visit(n_tree)

    n_code = ast.unparse(ast.fix_missing_locations(n_tree))

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
        new_ast = ast.parse(new_code)
        notrig_ast = ast.parse(new_code)

        if can_transform:
            for item in zip(patterns, pattern_tnames, can_transform_cache):
                pattern, pattern_tname, can_individual = item
                if can_individual and do_pattern:
                    new_ast = pattern.transform(new_ast, pattern_tname)
                    notrig_ast = pattern.transform(notrig_ast, pattern_tname)

            new_code = ast.unparse(ast.fix_missing_locations(new_ast))

        # for notrig code, we use the original docstring
        # notrig is only kept for compatibility, we dont have triggers here
        notrig_code = ast.unparse(ast.fix_missing_locations(notrig_ast))

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


def perform_code_transform(args):
    return _perform_code_transform_impl(*args)


def _check_transformability_impl(
    idx: int,
    obj: dict,
    patterns: List[code_manip.BaseTransformer],
    pattern_tnames: List[str],
    pattern_logic: str = "or",
):
    # get original docstring
    old_doc = obj["docstring"]

    # normalize code, remove old docstring
    code = remove_python_comments(obj["code"])
    n_tree = ast.parse(code)

    # inject original docstring to n_code
    n_tree = InsertDocstringVisitor(old_doc).visit(n_tree)

    # can_transform depends on logic_connective
    can_transform_cache = [p.can_transform(n_tree, t) for p, t in zip(patterns, pattern_tnames)]
    if pattern_logic == "and":
        can_transform = all(can_transform_cache)
    elif pattern_logic == "or":
        can_transform = any(can_transform_cache)
    else:
        raise ValueError(f"Unknown logic connective: {pattern_logic}")

    return (idx, can_transform)


def check_transformability(args):
    return _check_transformability_impl(*args)


def code_transformation_pipeline(
    objs: List[Dict],
    lang: str,
    split: str,
    patterns: List[code_manip.BaseTransformer],
    pattern_tnames: List[str],
    do_pattern: bool = True,
    target_num: Optional[int] = None,
    pattern_logic: str = "or",
    use_augmentation: bool = False,
    augment_kwargs: Optional[dict] = None,
):
    if target_num is None:
        if use_augmentation:
            raise ValueError("target_num must be specified when use_augmentation is True")

        print("target_num is None, using all samples")
        target_num = len(objs)

    if use_augmentation:
        # 1. check can transform
        transformability_args = []
        print("preparing for transformability scan")
        for idx, obj in enumerate(tqdm(objs)):
            transformability_args.append((idx, obj, patterns, pattern_tnames, pattern_logic))

        can_transforms = []
        with Pool(os.cpu_count() // 4) as pool:
            # [(idx, can_transform), ...]
            results = pool.imap(check_transformability, transformability_args)

            for idx, can_transform in tqdm(results, total=len(objs)):
                if can_transform:
                    can_transforms.append(idx)

        n_missing = target_num - len(can_transforms)
        print(f"detected {len(can_transforms)} transformable samples")
        print(f"expected: {target_num}")
        print(f"missing: {n_missing}")

        # 2. randomly select missing samples
        if split == "train" and n_missing > 0:
            population = set(range(len(objs))) - set(can_transforms)
            idx_to_augment = random.sample(list(population), n_missing)
        else:
            idx_to_augment = []
    else:
        idx_to_augment = []

    # 3. perform code transformation
    print("preparing for transformation")

    args = []
    idx_to_augment = set(idx_to_augment)
    for idx, obj in enumerate(tqdm(objs)):
        instance_id = f"{lang}#{split}#{idx+1}"
        args.append(
            (
                idx,
                instance_id,
                obj,
                patterns,
                pattern_tnames,
                do_pattern,
                pattern_logic,
                use_augmentation,
                idx_to_augment,
                augment_kwargs,
            )
        )

    n_ops = len(objs)
    n_good_ops = 0
    n_bad_ops = 0
    n_cant_ops = 0
    print(f"starting a total of {n_ops} transformations")

    transformed_instances: List[BackdoorCodeInstance] = []
    with Pool(os.cpu_count() // 4) as pool:
        results = pool.imap(perform_code_transform, args)

        for res in tqdm(results, total=n_ops):
            if res.error is not None:
                print(res.error)
                n_bad_ops += 1
                continue

            if res.can_transform:
                n_good_ops += 1
                transformed_instances.append(res)

                if n_good_ops < 5:
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

    return transformed_instances
