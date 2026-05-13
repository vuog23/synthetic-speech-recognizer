import torch
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

from src.data.datasets.dataset import MelSpectrogramDataset


class Evaluator:
    def __init__(
        self,
        model_path: str,
        weight_path: str,
        in_chans: int = 1,
        num_classes: int = 2,
        batch_size: int = 64,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            model_path,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )

        self.batch_size = batch_size

    def evaluate(self, test_path: str):

        test_loader = DataLoader(
            MelSpectrogramDataset(test_path),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for imgs, labels in test_loader:

                imgs = imgs.to(self.device)

                out = self.model(imgs)

                preds = out.argmax(dim=1).cpu()

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")
        mcc = matthews_corrcoef(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        cm = confusion_matrix(all_labels, all_preds)

        print("\nEvaluation Results")
        print("-" * 30)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"MCC      : {mcc:.4f}")
        print(f"Kappa    : {kappa:.4f}")

        print("\nConfusion Matrix")
        print(cm)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,
            "confusion_matrix": cm,
        }