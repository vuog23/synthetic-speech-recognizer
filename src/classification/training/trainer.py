import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
        drop_path_rate: float=0.1,

        batch_size: int=64,
        lr: float=1e-3,
        weight_decay: float=1e-5,
        epochs: int=30,

        num_workers: int = 2,
        use_amp: bool = True,
        ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.use_amp = use_amp and self.device.type == "cuda"

        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")

        self.preprocessor = AudioPreprocessor(
            sr=16000,
            target_length=48000,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            normalize_wav=False,
        )
        
        self.model = timm.create_model(
            model_path, # "convnext_tiny_hnf.a2h_in1k"
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

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
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
            for specs, labels in loader:
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

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):

                    outputs = self.model(specs)

                    loss = self.criterion(
                        outputs,
                        labels
                    )

                if train:
                    self.scaler.scale(
                        loss
                    ).backward()

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

                    self.scheduler.step()

                batch_size = specs.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += batch_size

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