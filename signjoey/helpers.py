# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
from torchtext.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False, resume: bool = False) -> str:
    """
    Create a new directory for the model, or reuse an existing one when
    resuming an interrupted run.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite (erase) an existing directory.
        Takes priority over `resume` if both are set.
    :param resume: if True and `model_dir` already exists, keep it as-is
        (checkpoints, logs, tensorboard files included) instead of raising
        or erasing it. Pass `resume=True` whenever `training.load_model`
        is set in the config -- that is exactly the "continue an
        interrupted/previous run" case, and erasing model_dir first would
        delete the very checkpoint you are about to load.
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if overwrite:
            # delete previous directory to start with empty dir again
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)
        elif resume:
            # keep everything (checkpoints, validations.txt, tensorboard
            # logs) in place; init_from_checkpoint (triggered by
            # training.load_model) will pick up training from here.
            return model_dir
        else:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled. If "
                "you are resuming an interrupted run, set training.load_model "
                "in your config -- this keeps the directory instead of "
                "erasing it. Otherwise, choose a new model_dir or set "
                "training.overwrite: true to start over."
            )
    else:
        os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu",weights_only=False)
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):

    """
    Create a link (or copy) from target to link_name.
    On Google Drive, symbolic links are not supported, so we use copy instead.
    """
    try:
        # Try symbolic link first (works on local filesystem)
        if os.path.islink(link_name):
            os.remove(link_name)
        os.symlink(target, link_name)
        return link_name
    except OSError:
        # If symlink fails (e.g., on Google Drive), use copy instead
        try:
            import shutil
            if os.path.exists(link_name):
                os.remove(link_name)
            shutil.copy2(target, link_name)
            return link_name
        except Exception as e:
            print(f"Warning: Could not create link/copy {link_name}: {e}")
            return None
        

def load_checkpoint_partial(
    model: nn.Module,
    path: str,
    use_cuda: bool = True,
    logger: Optional[Logger] = None,
) -> dict:
    """
    Load a checkpoint into `model` with strict=False, for cross-stage
    warm-starts where the architecture changed (e.g. Stage 1 batch+softsign
    -> Stage 2 layer+gelu): shapes that match transfer their weights,
    keys present only in the checkpoint (e.g. IntBatchNorm1d's
    running_mean/running_var) or only in the model (e.g. IntLayerNorm's
    lack thereof) are reported and skipped rather than raising.

    This is intentionally distinct from `helpers.load_checkpoint` +
    `model.load_state_dict` (used for same-architecture resume/QAT
    toggling, where strict=True is the correct safety net -- silently
    dropping a mismatched key there would hide a real bug). Cross-stage
    transitions are the ONE place a mismatch is expected and desired.

    :param model: freshly built target-stage model (e.g. Stage 2, layer+gelu)
    :param path: path to the SOURCE-stage checkpoint (e.g. Stage 1, batch+softsign)
    :param use_cuda: map_location for the checkpoint
    :param logger: optional logger; falls back to print if None
    :return: the raw checkpoint dict (steps/optimizer_state are NOT
        reused across stages -- optimizer/scheduler restart fresh,
        since the parameter set changed)
    """
    log = logger.info if logger is not None else print

    checkpoint = load_checkpoint(path=path, use_cuda=use_cuda)
    src_state = checkpoint["model_state"]
    tgt_state = model.state_dict()

    transferred, shape_mismatch, missing_in_src, unused_in_tgt = [], [], [], []

    for key, tgt_tensor in tgt_state.items():
        if key not in src_state:
            missing_in_src.append(key)
            continue
        src_tensor = src_state[key]
        if src_tensor.shape != tgt_tensor.shape:
            shape_mismatch.append(
                f"{key}: src{tuple(src_tensor.shape)} vs tgt{tuple(tgt_tensor.shape)}"
            )
            continue
        tgt_state[key] = src_tensor
        transferred.append(key)

    unused_in_tgt = [k for k in src_state.keys() if k not in tgt_state]

    model.load_state_dict(tgt_state)

    log(
        "Partial checkpoint load from %s:\n"
        "\tTransferred: %d tensors\n"
        "\tSkipped (shape mismatch): %d -> %s\n"
        "\tMissing in source (new tensors, kept at fresh init): %d -> %s\n"
        "\tUnused source tensors (architecture no longer has them): %d -> %s",
        path,
        len(transferred),
        len(shape_mismatch), shape_mismatch,
        len(missing_in_src), missing_in_src,
        len(unused_in_tgt), unused_in_tgt,
    )

    return checkpoint
