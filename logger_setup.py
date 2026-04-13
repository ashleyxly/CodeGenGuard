import os
import logging
from datetime import datetime


def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _setup_streaming_logger():
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)
    return logger


def _setup_file_logger(log_dir: str, filename: str):
    fpath = os.path.join(log_dir, filename)
    logger = logging.getLogger("generation_logger")
    logger.setLevel(logging.INFO)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(filename=fpath, mode="w", encoding="utf-8")
    file_formatter = logging.Formatter("%(message)s")

    # stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger, fpath


def setup_mixed_generation_logger(args):
    log_dir = args.logging_dir
    model_arch = args.model
    mixing_mode = args.mixing_mode
    pattern = "-".join(args.patterns)
    if args.no_soft_prompt:
        n_prefix_tokens = "ordinary"
    else:
        n_prefix_tokens = args.n_prefix_tokens
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-{model_arch}-{pattern}-{n_prefix_tokens}-{mixing_mode}"
    if args.base_checkpoint is not None:
        filename += "-ft"

    filename += ".log"
    return _setup_file_logger(log_dir, filename)


def setup_method_generation_logger(args):
    log_dir = args.output_dir
    # checkpoint = os.path.basename(args.checkpoint) if args.checkpoint is not None else "none"
    timestamp = _timestamp()
    filename = f"{timestamp}-gen.log"

    return _setup_multipurpose_logger(log_dir, filename)


def setup_generation_logger(args):
    log_dir = args.logging_dir
    pattern = args.pattern
    checkpoint = os.path.basename(args.checkpoint) if args.checkpoint is not None else "none"

    timestamp = _timestamp()

    filename = f"{timestamp}-gen-{pattern}-{checkpoint}"

    filename += ".log"
    return _setup_file_logger(log_dir, filename)


def setup_verification_logger(args, postfix: str = ""):
    log_dir = args.logging_dir
    pattern = args.pattern
    data_source = args.data_source

    name = f"verify-{data_source}"
    if postfix:
        name += f"-{postfix}"
    if args.random_prompt:
        name += "-random"

    timestamp = _timestamp()
    filename = f"{timestamp}-{name}-{pattern}"

    filename += ".log"
    return _setup_file_logger(log_dir, filename)


def setup_cross_generation_logger(args):
    log_dir = args.logging_dir
    model_arch = args.model
    pattern = args.pattern
    base_pattern = args.base_pattern
    if args.no_soft_prompt:
        n_prefix_tokens = "ordinary"
    else:
        n_prefix_tokens = args.n_prefix_tokens
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-{model_arch}-{base_pattern}-{pattern}-{n_prefix_tokens}"
    if args.base_checkpoint is not None:
        filename += "-ft"

    filename += ".log"
    return _setup_file_logger(log_dir, filename)


def _setup_multipurpose_logger(log_dir: str, filename: str):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, filename), mode="a", encoding="utf-8"
    )
    file_formatter = logging.Formatter("[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def setup_training_logger_spwm(args):
    model_arch = args.model
    pattern = args.pattern
    n_prefix_tokens = args.n_prefix_tokens
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-{model_arch}-{pattern}-{n_prefix_tokens}"
    if args.base_checkpoint is not None:
        filename += "-ft"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)


def setup_evaluation_logger(args):
    model_arch = args.model
    checkpoint = os.path.basename(args.checkpoint if args.checkpoint is not None else "none")
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-eval-{model_arch}-{checkpoint}"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)


def setup_extraction_logger_peftwm(args):
    model_arch = args.student_model
    teacher_checkpoint = args.teacher_checkpoint
    teacher_name = os.path.basename(teacher_checkpoint)
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-extract-{model_arch}-{teacher_name}"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)


def setup_finetune_removal_logger(args):
    model_name = args.model_type
    log_dir = args.logging_dir

    timestamp = _timestamp()

    filename = f"{timestamp}-rm_ft-{model_name}"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)


def setup_training_logger_peftwm(args):
    model_arch = args.model
    pattern = args.pattern
    log_dir = args.logging_dir

    timestamp = _timestamp()
    train_script_name = args.script_name
    script_main_name = os.path.splitext(os.path.basename(train_script_name))[0]
    script_main_name = script_main_name[script_main_name.find("_") + 1 :]

    filename = f"{timestamp}-{script_main_name}-{model_arch}-{pattern}"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)


def setup_training_logger_multiwm(args):
    model_arch = args.model
    pattern = args.pattern
    n_prefix_tokens = args.n_prefix_tokens
    log_dir = args.logging_dir
    n_clients = args.n_local_clients

    timestamp = _timestamp()

    filename = f"{timestamp}-{model_arch}-{pattern}-{n_prefix_tokens}-{n_clients}"

    filename += ".log"
    return _setup_multipurpose_logger(log_dir, filename)
