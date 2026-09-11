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
    roc_auc_score,
)


from src.classification.data.for_data import FoRDataset
from src.classification.features.mel_spectrogram import AudioPreprocessor


class Evaluator:

    def __init__(
        self,
        model_path: str,
        weight_path: str,
        in_chans: int = 1,
        num_classes: int = 2,
        batch_size: int = 32,
        num_workers: int = 2,
    ):


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.preprocessor = AudioPreprocessor(
            sr=16000,
            target_length=48000,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            normalize_wav=False,
        )

        self.model = timm.create_model(
            model_path,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes,
        ).to(self.device)

        state_dict = torch.load(
            weight_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)

        self.model.eval()

        print(f"Evaluator device: {self.device}")
        print(f"Loaded weights: {weight_path}")


    def evaluate(
        self,
        data_root: str,
        split: str = "test",
    ):

        test_dataset = FoRDataset(
            root_path=data_root,
            split=split,
            preprocessor=self.preprocessor,
            augment=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.inference_mode():

            for specs, labels in test_loader:

                specs = specs.to(
                    self.device,
                    non_blocking=True,
                )

                outputs = self.model(specs)

                probs = torch.softmax(
                    outputs,
                    dim=1,
                )

                fake_probs = probs[:, 1]

                preds = outputs.argmax(
                    dim=1
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    labels.numpy()
                )

                all_probs.extend(
                    fake_probs.cpu().numpy()
                )

        acc = accuracy_score(
            all_labels,
            all_preds,
        )

        precision = precision_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0,
        )

        recall = recall_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0,
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0,
        )

        mcc = matthews_corrcoef(
            all_labels,
            all_preds,
        )

        kappa = cohen_kappa_score(
            all_labels,
            all_preds,
        )

        cm = confusion_matrix(
            all_labels,
            all_preds,
        )

        auc = roc_auc_score(
                all_labels,
                all_probs,
            )

        print()
        print("Evaluation Results")
        print("-" * 40)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"MCC      : {mcc:.4f}")
        print(f"Kappa    : {kappa:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")

        print()
        print("Confusion Matrix")
        print(cm)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,
            "roc_auc": auc,
            "confusion_matrix": cm,
        }