import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd

from src.classification.data.for_data import FoRDataset
from src.classification.features.mel_spectrogram import AudioPreprocessor

class Trainer:
    def __init__(
        self,
        model_path: str,
        data_root: str,

        in_chans: int=1,
        num_classes: int=2,
        drop_path_rate: float=0.0,

        batch_size: int=64,
        lr: float=1e-4,
        epochs: int=30,

        num_workers: int = 2,
        max_grad_norm: float = 1.0,
        use_class_weights: bool = True,
        check_finite: bool = True,
        ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.check_finite = check_finite

        print(f"Device: {self.device}")
        print("AMP: False (FP32 training)")

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
            pretrained=True,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        ).to(self.device)

        
        self.train_dataset = FoRDataset(
            root_path=data_root,
            split="train",
            preprocessor=self.preprocessor,
            augment=True,
        )

        self.val_dataset = FoRDataset(
            root_path=data_root,
            split="val",
            preprocessor=self.preprocessor,
            augment=False,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
        )

        if use_class_weights:
            labels = torch.tensor(
                [label for _, label in self.train_dataset.samples],
                dtype=torch.long,
            )
            class_counts = torch.bincount(
                labels,
                minlength=num_classes,
            ).float()

            if (class_counts == 0).any():
                raise RuntimeError(
                    f"Every class must contain samples; counts={class_counts.tolist()}"
                )

            class_weights = class_counts.sum() / (
                num_classes * class_counts
            )
            class_weights = class_weights.to(self.device)

            print(f"Class counts: {class_counts.tolist()}")
            print(f"Class weights: {class_weights.tolist()}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )


    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool
    ):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        if train:
            context = torch.enable_grad()
        else:
            context = torch.inference_mode()

        with context:
            for batch_index, (specs, labels) in enumerate(loader):

                if self.check_finite and not torch.isfinite(specs).all():
                    phase = "train" if train else "validation"
                    raise FloatingPointError(
                        f"Non-finite input in {phase} batch {batch_index}."
                    )

                specs = specs.to(
                    self.device,
                    non_blocking=True
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True
                )

                if train:
                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                outputs = self.model(specs)

                if self.check_finite and not torch.isfinite(outputs).all():
                    phase = "train" if train else "validation"
                    raise FloatingPointError(
                        f"Non-finite model output in {phase} batch "
                        f"{batch_index}; lr={self.optimizer.param_groups[0]['lr']:.3e}."
                    )

                # Loss
                loss = self.criterion(
                    outputs,
                    labels
                )

                if self.check_finite and not torch.isfinite(loss):
                    phase = "train" if train else "validation"
                    raise FloatingPointError(
                        f"Non-finite loss in {phase} batch {batch_index}; "
                        f"lr={self.optimizer.param_groups[0]['lr']:.3e}."
                    )

                if train:
                    loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )

                    if self.check_finite and not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm in train batch "
                            f"{batch_index}; lr="
                            f"{self.optimizer.param_groups[0]['lr']:.3e}."
                        )

                    self.optimizer.step()

                current_batch_size = specs.size(0)

                total_loss += (
                    loss.item() * current_batch_size
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += current_batch_size

        if total == 0:
            raise RuntimeError(
                "DataLoader contains zero samples."
            )

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy


    def train(
        self,
        save_path: str = "best_model.pt",
        history_path: str = "history.csv",
    ):

        best_val_loss = float("inf")

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        header = (
            f"{'Epoch':>6} | "
            f"{'Train Loss':>10} | "
            f"{'Train Acc':>9} | "
            f"{'Val Loss':>9} | "
            f"{'Val Acc':>9} | "
            f"{'Status':>10}"
        )

        divider = "-" * len(header)

        print()
        print(
            f"Training for up to {self.epochs} "
            f"epochs on {self.device}"
        )

        print(divider)
        print(header)
        print(divider)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(
                self.train_loader,
                train=True
            )

            val_loss, val_acc = self._run_epoch(
                self.val_loader,
                train=False
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            status = ""

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    save_path
                )

                status = "best model"

            print(
                f"{epoch:>6} | "
                f"{train_loss:>10.4f} | "
                f"{train_acc:>8.2%} | "
                f"{val_loss:>9.4f} | "
                f"{val_acc:>8.2%} | "
                f"{status:>10}"
            )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        print()
        print(
            f"History saved to '{history_path}'"
        )

        print(
            f"Best val loss: {best_val_loss:.4f}"
        )
