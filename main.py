import torch
import numpy as np
import argparse
import os
from dynaconf import settings

from src.config import RobertaConfig
from src.models import RobertaClassificationModel
from src.preprocessing.classification import IMDB_LABEL_TO_IDX
from src.utils.dataset import get_classify_dataset
from src.utils.optim import get_optimizer, get_group_params, get_linear_scheduler_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default="imdb")
    parser.add_argument("--gradient_accumulation_steps", default=2)
    parser.add_argument("--batches_per_epoch", default=0)
    parser.add_argument("--epochs", default=40)

    parser.add_argument("--optim_method", default="adamw")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1., type=float)
    return parser.parse_args()

def compute_warmup_steps(
        args: argparse.Namespace,
        warmup_persentage: float = 1.5
) -> argparse.Namespace:
    args.steps_per_epoch = int(args.batches_per_epoch / args.gradient_accumulation_steps)
    args.num_warmup_steps = args.steps_per_epoch * warmup_persentage
    args.num_training_steps = int(args.steps_per_epoch * args.epochs)
    return args

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    device = torch.device(args.device)

    conf = RobertaConfig(num_labels=len(IMDB_LABEL_TO_IDX))
    model = RobertaClassificationModel.load(
        conf,
        os.path.join(
            settings.get("exp_dir"),
            "pre-trained",
            "roberta-base.pth.tar"
        ),
        mode="pre-trained"
    )

    group_params = get_group_params(
        model.named_parameters(),
        args.weight_decay,
        no_decay=["bias", "layer_norm.weight", "layer_norm.bias"]
    )

    # model = torch.nn.DataParallel(model, device_ids=[1, 0])
    model = model.to(device)

    train_epoch_gen = get_classify_dataset(
        os.path.join(settings.get("data_dir"), "classification", f"{args.dataset}.pt"),
        "train",
        max_tokens_per_batch=4000,
        pad=conf.pad_idx,
        max_sentence_length=512
    )

    eval_epoch_gen = get_classify_dataset(
        os.path.join(settings.get("data_dir"), "classification", f"{args.dataset}.pt"),
        "eval",
        max_tokens_per_batch=4000,
        pad=conf.pad_idx,
        max_sentence_length=512
    )
    if args.batches_per_epoch <= 0:
        args.batches_per_epoch = len(train_epoch_gen)
    args = compute_warmup_steps(args)

    optim = get_optimizer(
        method=args.optim_method,
        params=group_params,
        lr=args.lr
    )
    scheduler = get_linear_scheduler_with_warmup(
        optim,
        args.num_warmup_steps,
        args.num_training_steps
    )