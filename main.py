import torch
import numpy as np
import argparse
import os
from dynaconf import settings
from datetime import datetime

from src.config import RobertaConfig
from src.models import RobertaClassificationModel
from src.preprocessing.classification import IMDB_LABEL_TO_IDX
from src.utils.dataset import get_classify_dataset
from src.utils.optim import get_optimizer, get_group_params, get_linear_scheduler_with_warmup, unfreeze_layer_params
from src.utils import save_data_to_json, save_checkpoint
from src.tasks import ClassificationTask

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="0")
    parser.add_argument("--model_version", default="0")
    parser.add_argument("--db_name", default="imdb")
    parser.add_argument("--db_version", default="0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_gpus", default=1)


    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--batches_per_epoch", default=0, type=int)
    parser.add_argument("--max_sentences_per_batch", default=50, type=int)
    parser.add_argument("--max_tokens_per_batch", default=7000, type=int)
    parser.add_argument("--max_sentence_length", default=1400, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--optim_method", default="adamw")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1., type=float)
    parser.add_argument("--eval_every", default=3, type=int)
    parser.add_argument("--epochs", default=40, type=int)
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

    conf = RobertaConfig(
        pos_embeddings_type="partually_fixed",
        max_position_embeddings=3002,
        fixed_num_embeddings=514,
        trainable_num_embeddings=3002-514,
        attention_type="linear",
        num_labels=len(IMDB_LABEL_TO_IDX)
    )

    model = RobertaClassificationModel.load(
        conf,
        os.path.join(
            settings.get("exp_dir"),
            "pre-trained",
            "roberta-import.pth.tar"
        ),
        mode="pre-trained"
    )

    group_params = get_group_params(
        model.named_parameters(),
        args.weight_decay,
        no_decay=["bias", "layer_norm.weight", "layer_norm.bias"]
    )

    model_name = model.name + args.model_version
    unfreeze_layer_params(model.named_parameters(), layer=3)
    # adjust positional embedding
    model.embed_positions.weight.requires_grad = True

    if torch.cuda.device_count() > 1 and args.n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[1, 0])
        args.max_sentences_per_batch *= args.max_sentences_per_batch
        args.max_tokens_per_batch *= (args.n_gpus // 2)

    model = model.to(device)

    train_epoch_gen = get_classify_dataset(
        os.path.join(settings.get("data_dir"), "classification", f"{args.dataset}.pt"),
        "train",
        max_tokens_per_batch=4000,
        pad_token_id=conf.pad_token_id,
        max_sentence_length=512
    )

    eval_epoch_gen = get_classify_dataset(
        os.path.join(settings.get("data_dir"), "classification", f"{args.dataset}.pt"),
        "eval",
        max_tokens_per_batch=4000,
        pad_token_id=conf.pad_token_id,
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

    exp_name = f'exp-{args.task_name}-{args.db_name}-{args.db_version}-{model_name}-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'
    print(f"RUNNING EXPERIMENT -> {exp_name}")

    conf.save(os.path.join(
        settings.get("ckp_dir"),
        args.task_name,
        f"{args.db_name}_{args.db_version}",
        model_name
    ))

    save_data_to_json(
        os.path.join(
            settings.get("ckp_dir"),
            args.task_name,
            f"{args.db_name}_{args.db_version}",
            model_name,
            "args.json"
        ),
        args.__dict__
    )

    best_f1 = 0.
    train_loss = []
    train_acc = []

    eval_loss = []
    eval_prec = []
    eval_recal = []
    eval_f1 = []
    eval_acc = []

    task = ClassificationTask(
        name=exp_name,
        args=args,
        pad_token_id=conf.pad_token_id,
        eos_token_id=conf.eos_token_id
    )

    for epoch in range(1, args.epochs + 1):
        train_iter_dl = train_epoch_gen.next_epoch_itr(shuffle=True)
        loss, acc = task.train(
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            dataloader=train_iter_dl,
            device=device
        )

        train_loss.append(loss)
        train_acc.append(acc)
        print(f"epoch:{epoch}\tacc:{acc}\tloss{loss}")
        task.tb_plot_scalar("train", dict(loss=loss, accuracy=acc), step=epoch - 1)

        if epoch % args.eval_every == 0 or epoch == 1:
            is_best = False
            eval_iter_dl = eval_epoch_gen.next_epoch_itr(shiffle=True)
            loss, (acc, prec, rec, f_score), _ = task.eval(
                model=model,
                dataloader=eval_iter_dl,
                device=device
            )

            eval_loss.append(loss)
            eval_acc.append(acc)
            eval_prec.append(prec)
            eval_recal.append(rec)
            eval_f1.append(f_score)
            print(f"--------->eval\tacc:{acc}\tloss{loss}\tprec:{prec}\trec:{rec}\tf1:{f_score}")
            task.tb_plot_scalar("eval",
                                dict(
                                    loss=loss,
                                    accuracy=acc,
                                    precision=prec,
                                    recal=rec,
                                    f_1=f_score
                                ),
                                step=epoch - 1
                                )

            if f_score > best_f1:
                best_f1 = f_score
                is_best = True

            if isinstance(model, torch.nn.DataParallel):
                state_dict = dict([(n, p.to("cpu")) for n, p in model.module.state_dict().items()])
            else:
                state_dict = dict([(n, p.to("cpu")) for n, p in model.state_dict().items()])

            save_checkpoint(
                path_=os.path.join(
                    settings.get("ckp_dir"),
                    args.task_name,
                    f"{args.db_name}_{args.db_version}",
                    model_name
                ),
                state=state_dict,
                is_best=is_best,
                filename=f"ckp_{epoch}.pth.tar"
            )