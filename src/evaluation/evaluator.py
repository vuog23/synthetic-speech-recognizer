import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    SwinForImageClassification,
    AutoImageProcessor,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

class _TestDataset(Dataset):
    def __init__(self, pt_path: str, image_processor):
        data        = torch.load(pt_path, map_location="cpu")
        self.specs  = data["specs"].float()   # (N, 1, 224, 224)
        self.labels = data["labels"].long()   # (N,)
        self.proc   = image_processor
        print(f"  Loaded {pt_path}: {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec   = self.specs[idx]                       # (1, 224, 224) in [0, 1]
        spec   = spec.repeat(3, 1, 1)                  # (3, 224, 224)
        np_img = spec.permute(1, 2, 0).numpy()         # (224, 224, 3)
        processed = self.proc(
            images         = np_img,
            return_tensors = "pt",
            do_rescale     = False,
        )
        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "label":        self.labels[idx],
        }


class Evaluator:
    def __init__(
        self,
        model_dir  : str,
        test_pt    : str,
        model_type : str = "vit",
        batch_size : int = 32,
        device     : str = None,
    ):
        self.model_dir  = model_dir
        self.test_pt    = test_pt
        self.model_type = model_type.lower()
        self.batch_size = batch_size
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.model_type == "vit":
            self.processor = ViTImageProcessor.from_pretrained(model_dir)
            self.model     = ViTForImageClassification.from_pretrained(model_dir)
        elif self.model_type == "swin":
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
            self.model     = SwinForImageClassification.from_pretrained(model_dir)
        else:
            raise ValueError(f"model_type must be 'vit' or 'swin', got '{model_type}'")

        self.model.to(self.device).eval()
        print(f"  Loaded {self.model_type.upper()} model from {model_dir}")
        print(f"  Running on: {self.device}")

        self.id2label = self.model.config.id2label   # {0: "fake", 1: "real"}

        dataset        = _TestDataset(test_pt, self.processor)
        self.loader    = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 2,
            pin_memory  = True,
        )

        self.all_labels   = None
        self.all_preds    = None
        self.all_probs    = None
        self.metrics      = None

    # ── Inference ─────────────────────────────────────────────────────────
    def _predict(self):
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for batch in self.loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels       = batch["label"]

                logits = self.model(pixel_values=pixel_values).logits   # (B, 2)
                probs  = torch.softmax(logits, dim=-1)                  # (B, 2)
                preds  = logits.argmax(-1)                              # (B,)

                all_labels.append(labels.numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        self.all_labels = np.concatenate(all_labels)
        self.all_preds  = np.concatenate(all_preds)
        self.all_probs  = np.concatenate(all_probs)   # (N, 2)

    # ── Metrics ───────────────────────────────────────────────────────────
    def _compute_metrics(self):
        y_true     = self.all_labels
        y_pred     = self.all_preds
        y_prob_pos = self.all_probs[:, 1]

        self.metrics = {
            "accuracy"  : accuracy_score(y_true, y_pred),
            "precision" : precision_score(y_true, y_pred, zero_division=0),
            "recall"    : recall_score(y_true, y_pred, zero_division=0),
            "f1"        : f1_score(y_true, y_pred, zero_division=0),
            "kappa"     : cohen_kappa_score(y_true, y_pred),
            "mcc"       : matthews_corrcoef(y_true, y_pred),
            "auc_roc"   : roc_auc_score(y_true, y_prob_pos),
        }

        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_prob_pos)
        self.confusion                           = confusion_matrix(y_true, y_pred)


    def run(self) -> dict:
        print("\n  Running inference …")
        self._predict()
        self._compute_metrics()
        self._print_results()
        return self.metrics

    def _print_results(self):
        label_names = [self.id2label[i] for i in sorted(self.id2label)]
        print("\n" + "─" * 45)
        print(f"  {'Metric':<18}{'Value':>10}")
        print("─" * 45)
        for name, val in self.metrics.items():
            print(f"  {name:<18}{val:>10.4f}")
        print("─" * 45)
        print(f"\n  Class mapping: {self.id2label}")
        print(f"  Confusion matrix (rows=true, cols=pred):")
        print(f"  Labels: {label_names}")
        print(f"  {self.confusion}")