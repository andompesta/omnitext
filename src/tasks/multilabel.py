import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, Tensor, optim
from typing import Tuple, Optional
from src.tasks import OmniTask
from src.utils.data import OmniDataset
from argparse import Namespace


class MultilabelClassificationTask(OmniTask):
    def __init__(
            self,
            name: str,
            args: Namespace,
            pad_token_id: int,
            eos_token_id: Optional[int],
            ignore_index: int = -1
    ):
        super(MultilabelClassificationTask, self).__init__(name, args)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.ignore_index = ignore_index

    @classmethod
    def get_loss_fn(
            cls,
            type: str = "binary_cross_entropy",
            reduction: str = 'none',
            pos_weight: Optional[Tensor] = None
    ):
        if type == "binary_cross_entropy":
            return nn.BCEWithLogitsLoss(
                reduction=reduction,
                pos_weight=pos_weight
            )
        else:
            raise NotImplementedError(f"loss {type} not yet implemented")


    @classmethod
    def compute_correct(
            cls,
            logits: Tensor,
            labels: Tensor,
            th: float = 0.5,
            **kwargs
    ) -> Tuple[Tensor, int]:
        with torch.no_grad():
            preds = torch.sigmoid(logits) > th
            preds = preds.long()
            n_correct = preds.eq(labels).sum().item()
            return preds, n_correct


    def train(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler.LambdaLR,
            dataloader: OmniDataset,
            device,
            **kwargs
    ) -> Tuple[float, float]:
        model.train()
        loss_fn = self.get_loss_fn(
            type=kwargs.get("loss_type", "binary_cross_entropy")
        )

        total_loss = 0
        n_pred_total = 0
        n_pred_correct = 0
        steps = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            src_seq_t, pos_seq_t, label_t = batch

            # compute mask
            mask = label_t != self.ignore_index
            label_t.masked_fill_(torch.logical_not(mask), 0.)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits_t = model(src_seq_t)
                loss_t = loss_fn(logits_t, label_t.float())
                loss_t *= mask.float()
                loss_t = loss_t.mean(0).sum(0)

                if self.args.gradient_accumulation_steps > 1:
                    # scale the loss if gradient accumulation is used
                    loss_t = loss_t / self.args.gradient_accumulation_steps

                loss_t.backward()
                nn.utils.clip_grad_norm(model.parameters(), self.args.max_grad_norm)

                # accumulate the gradients
                if batch_idx % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()

            # update metrics
            steps += 1
            pred_t, n_correct = self.compute_correct(logits_t, label_t)
            total_loss += loss_t.item()
            n_pred_total += label_t.size(0)
            n_pred_correct += n_correct

            if batch_idx % 100:
                torch.cuda.empty_cache()
                print(f"batch : {batch_idx}")

            if (steps / self.args.gradient_accumulation_steps) == self.args.steps_per_epoch:
                break

        steps /= self.args.gradient_accumulation_steps
        total_loss = total_loss / steps
        accuracy = n_pred_correct / n_pred_total
        self.global_step += int(steps)
        return total_loss, accuracy

    def eval(
            self,
            model: nn.Module,
            dataloader: OmniDataset,
            device,
            **kwargs
    ):
        model.eval()

        loss_fn = self.get_loss_fn(
            type=kwargs.get("loss_type", "binary_cross_entropy")
        )
        total_loss = 0
        n_pred_total = 0
        n_pred_correct = 0
        steps = 0

        preds = []
        labels = []

        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            src_seq_t, labels_t = batch

            # compute mask
            mask = labels_t != self.ignore_index
            labels_t.masked_fill_(1 - mask, 0.)

            with torch.set_grad_enabled(False):
                logits_t = model(src_seq_t)
                loss_t = loss_fn(logits_t, labels_t.float())
                loss_t *= mask
                loss_t = loss_t.mean(0).sum(0)

            pred_t, n_correct = self.compute_correct(logits_t, labels_t)
            preds.append(pred_t.detach_().cpu().numpy())
            labels.append(labels_t.detach_().cpu().numpy())

            total_loss += loss_t
            n_pred_total += labels_t.size(0)
            n_pred_correct += n_correct
            steps += 1

        total_loss /= steps
        accuracy = n_pred_correct / n_pred_total

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)

        prec, rec, f_score, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro"
        )

        scores = dict(
            loss=total_loss,
            accuracy=accuracy,
            precision=prec,
            recall=rec,
            f_score=f_score
        )
        return scores