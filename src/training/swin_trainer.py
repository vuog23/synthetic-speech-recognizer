import torch
from transformers import Trainer
import os
from src.config.swin_config import SwinConfig

class SwinTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        compute_metrics,
        callbacks,
    ):
        super().__init__(
            model           = model,
            args            = args,
            train_dataset   = train_dataset,
            eval_dataset    = eval_dataset,
            compute_metrics = compute_metrics,
            callbacks       = callbacks,
        )
        self._epoch_correct = 0
        self._epoch_total   = 0
 
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = SwinConfig.LR,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
            weight_decay = 0.01,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr           = SwinConfig.LR,
            total_steps      = num_training_steps,
            pct_start        = 0.1,
            anneal_strategy  = "cos",
            div_factor       = 25.0, 
            final_div_factor = 1e4, 
        )
 
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs      : bool = False,
        num_items_in_batch  : int  = None,
    ):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        loss    = outputs.loss
 
        if model.training and labels is not None:
            preds = outputs.logits.detach().argmax(-1)
            self._epoch_correct += (preds == labels.detach()).sum().item()
            self._epoch_total   += labels.size(0)
 
        return (loss, outputs) if return_outputs else loss
 
    def evaluate(
        self,
        eval_dataset      = None,
        ignore_keys       = None,
        metric_key_prefix : str = "eval",
    ):
        train_acc = (
            self._epoch_correct / self._epoch_total
            if self._epoch_total > 0 else 0.0
        )
        self._epoch_correct = 0
        self._epoch_total   = 0
 
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics["train_acc"] = train_acc
        return metrics
 
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)