import functools
import json
import logging
import math
from pathlib import Path
from typing import Callable, Union

import safetensors
import torch
# Remove FSDP imports
# import torch.distributed.fsdp.wrap as torch_wrap
# from torch.distributed.fsdp import BackwardPrefetch
# from torch.distributed.fsdp.api import ShardingStrategy
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from model.args import ModelArgs, MoeArgs
from model.transformer import Transformer, TransformerBlock

from .args import LoraArgs
from .checkpointing import Checkpointer
from .distributed import (
    get_rank,
    get_world_size,
)

logger = logging.getLogger(__name__)

# This function is no longer needed without distributed support
# def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
#     # Logic for FSDP wrapping is not needed on CPU
#     return None

def log_train_params(model: torch.nn.Module):
    # Remove world size and distributed logic
    num_params = sum(p.numel() for p in model.parameters())
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )

def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError("Only Lora layers should be randomly initialized.")

def load_args(folder: Path, lora: LoraArgs) -> ModelArgs:
    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())

    model_args = ModelArgs(
        lora=lora,
        dim=args["dim"],
        n_layers=args["n_layers"],
        head_dim=args["head_dim"],
        hidden_dim=args["hidden_dim"],
        n_heads=args["n_heads"],
        n_kv_heads=args["n_kv_heads"],
        norm_eps=args["norm_eps"],
        vocab_size=args["vocab_size"],
    )

    if args.get("rope_theta") is not None:
        model_args.rope_theta = args["rope_theta"]

    if args.get("moe") is not None:
        model_args.moe = MoeArgs(**args["moe"])

    return model_args

def load_model(
    folder: Path,
    lora: LoraArgs,
    checkpoint: bool,
    param_dtype: torch.dtype,
) -> torch.nn.Module:  # Updated return type to torch.nn.Module
    model_args = load_args(folder, lora)

    if model_args.vocab_size == 32000:
        raise ValueError(
            f"Fine-tuning is not supported for older model versions with vocab_size 32000. Make sure to extend your model to vocab_size=32768 using `python -m utils.extend_model_vocab --original_model_ckpt {folder} --extended_model_ckpt {folder}_extended`."
        )

    assert (
        model_args.vocab_size >= 32768
    ), "Make sure to use a model with a vocab size of at least 32768"

    # Load model on CPU
    with torch.device("cpu"):
        model = Transformer(args=model_args, checkpoint=checkpoint)

    # Load model state
    state_dict = load_state_dict(folder, dtype=param_dtype)
    model.load_state_dict(state_dict, assign=True)

    logger.info("Loaded model on CPU!")

    if lora.enable:
        logger.info("Initializing lora layers ...")
        initialize_lora_parameters(model, param_dtype)

    # Verify all parameters are initialized
    assert not any(
        p.is_meta for p in model.parameters()
    ), "All parameters should be initialized by now"
    assert all(
        p.dtype == param_dtype for p in model.parameters()
    ), f"All parameters should be on {param_dtype}"

    logger.info("Finished initialization!")

    # Freeze non-LoRA parameters
    if lora.enable:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    log_train_params(model)

    return model  # Return the regular PyTorch model without FSDP wrapping

@torch.no_grad()
def load_state_dict(path: Path, dtype: torch.dtype):
    assert path.is_dir(), path

    this_safetensors_path = Checkpointer.consolidated_path(path, use_safetensors=True)
    this_torch_path = Checkpointer.consolidated_path(path, use_safetensors=False)

    assert (
        this_safetensors_path.exists() or this_torch_path.exists()
    ), f"Either {this_safetensors_path} or {this_torch_path} must exist."
    assert not (
        this_safetensors_path.exists() and this_torch_path.exists()
    ), f"Only one of {this_safetensors_path} or {this_torch_path} should exist."

    if this_safetensors_path.exists():
        logger.info(f"Reloading model from {this_safetensors_path} ...")
        model_state_dict = safetensors.torch.load_file(this_safetensors_path)
    else:
        logger.info(f"Reloading model from {this_torch_path} ...")
        model_state_dict = torch.load(this_torch_path)

    logger.info(f"Converting model to dtype {dtype} ...")

    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(dtype)

    return model_state_dict