import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, Tensor, optim
from typing import Tuple
from src.tasks import OmniTask
from src.utils.data import OmniDataset

class ClassificationTask(OmniTask):
    def __init__(
            self,
            name: str,
            args,
            global_step: int = 0,
            pad_idx: int = 1
    ):
        super(ClassificationTask, self).__init__(name, args, global_step)
        self.pad_idx = pad_idx

    def get_loss_fn(self, pad_idx: int, reduction='none'):
        return nn.CrossEntropyLoss(
            reduction=reduction,
            ignore_index=pad_idx
        )

    def compute_correct(self, logits: Tensor, labels: Tensor, **kwargs) -> Tuple[Tensor, int]:
        with torch.no_grad():
            pred_idx = logits.argmax(1)
            n_correct = pred_idx.eq(labels).sum().item()
            return pred_idx, n_correct


    def train(
            self,
            model: nn.Module,
            optimizer: optim.optimizer.Optimizer,
            scheduler: optim.lr_scheduler.LambdaLR,
            dataloader: OmniDataset,
            device,
            **kwargs
    ) -> Tuple[float, float]:
        model.train()
        optimizer.zero_grad()
        loss_fn = self.get_loss_fn(self.pad_idx)

        total_loss = 0
        n_pred_total = 0
        n_pred_correct = 0
        steps = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            src_seq_t, label_t = batch

            with torch.set_grad_enabled(True):
                logits_t, *_ = model(src_seq_t)
                loss_t = loss_fn(logits_t, label_t)
                loss_t = loss_t.mean(-1)

                if self.args.gradient_accumulation_steps > 1:
                    # scale the loss if gradient accumulation is used
                    loss_t = loss_t / self.args.gradient_accumulation_steps

                loss_t.backward()
                nn.utils.clip_grad_norm(model.parameters(), self.args.max_grad_norm)

                # accumulate the gradients
                if batch_idx % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # update metrics
            steps += 1
            pred_t, n_correct = self.compute_correct(logits_t, label_t)
            total_loss += loss_t.item()
            n_pred_total += label_t.size(0)
            n_pred_correct += n_correct

            if batch_idx % 100:
                torch.cuda.empty_cache()
                print(f"batch : {batch_idx}")

            # TODO: add gradual unfreeze

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

        loss_fn = self.get_loss_fn(self.pad_idx)
        total_loss = 0
        n_pred_total = 0
        n_pred_correct = 0
        steps = 0

        preds = []
        labels = []

        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            src_seq_t, labels_t = batch

            with torch.set_grad_enabled(False):
                logits_t, *_ = model(src_seq_t)
                loss_t = loss_fn(logits_t, labels_t).mean(-1).item()

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