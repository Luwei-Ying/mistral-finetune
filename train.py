import dataclasses
import logging
import os
import pprint
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
)
from finetune.wrapped_model import load_model, load_args

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    print(f"args: {args}")
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(
    args: TrainArgs,
    exit_stack: ExitStack,
):
    # 1. Initial setup and checks
    set_random_seed(args.seed)

    # Ensure we are using the CPU
    device = torch.device("cpu")
    logger.info(f"Running on device: {device}")

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Initialize the logger without wandb
    metrics_logger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=True,
        # wandb_args=None,  # Skip wandb entirely
        mlflow_args=args.mlflow,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=True,
        # wandb_args=None,  # Skip wandb here as well
        mlflow_args=args.mlflow,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 5. Potentially download model
    if Path(args.model_id_or_path).is_dir():
        model_folder = Path(args.model_id_or_path)
    else:
        raise ValueError(
            "Invalid folder path. Please set `args.initial_model` to a valid folder path."
        )

    # 6. Load function calling instruct tokenizer
    vocab_size = load_args(model_folder, args.lora).vocab_size
    is_tekken = vocab_size > 32768

    instruct_tokenizer: InstructTokenizerBase = MistralTokenizer.v3(
        is_tekken=is_tekken
    ).instruct_tokenizer  # type: ignore

    # 7. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=0,  # Single process, so rank is always 0
        world_size=1,  # Single process, so world_size is always 1
        is_eval=False,
    )

    if not args.no_eval:
        assert (
            args.data.eval_instruct_data != ""
        ), "Either set `no_eval` to True or provide evaluation samples under `data.eval_instruct_data`"

        eval_data_loader = build_data_loader(
            instruct_tokenizer=instruct_tokenizer,
            args=args.data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=None,
            rank=0,
            world_size=1,
            is_eval=True,
        )
        eval_batches = list(eval_data_loader)

    # 8. Load model
    param_dtype = torch.float32  # Use float32 for CPU

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    model = load_model(
        folder=model_folder,
        lora=args.lora,
        checkpoint=args.checkpoint,
        param_dtype=param_dtype,
    ).to(device)  # Ensure model is loaded to CPU

    # 9. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 10. Initialize checkpointer
    checkpointer = Checkpointer(
        model=model,
        state=state,
        run_dir=run_dir,
        optimizer=optimizer,
        num_ckpt_keep=args.num_ckpt_keep
    )

    model.train()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device=device)
        n_batch_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)

            x = torch.from_numpy(batch.x).to(device)
            y = torch.from_numpy(batch.y).to(device)
            y_mask = (
                torch.from_numpy(batch.y_mask).to(device)
                if batch.y_mask is not None
                else None
            )

            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )
            mb_loss = compute_loss_with_mask(output, y, y_mask)

            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += x.numel()

        loss /= args.num_microbatches

        optimizer.step()

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = loss_item  # Single process, no need for avg_aggregate

        if not args.no_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            evaluate(model, eval_batches, state)

        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                last_lr,
                0,  # No GPU, so no memory tracking
                0,  # No GPU, so no memory tracking
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=args.save_adapters,
                dtype=param_dtype,
                instruct_tokenizer=instruct_tokenizer,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    fire.Fire(train)