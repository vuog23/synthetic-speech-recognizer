import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data.datasets.dataset import MelSpectrogramDataset
from src.model.architecture.madnet import MADNet
import pandas as pd

class Trainer:
    def __init__(
        self,
        model        : MADNet,
        train_path   : str,
        val_path     : str,
        num_classes  : int   = 2,
        batch_size   : int   = 32,
        lr           : float = 1e-3,
        weight_decay : float = 1e-5,
        epochs       : int   = 100,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = model.to(self.device)
        self.epochs = epochs

        self.train_loader = DataLoader(
            MelSpectrogramDataset(r"/kaggle/input/datasets/trieuvuongnguyen/for-preprocessed/processed/train.pt"),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            MelSpectrogramDataset(r"/kaggle/input/datasets/trieuvuongnguyen/for-preprocessed/processed/val.pt"),
            batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            drop_last=False,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr * 10,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )

    def _train_epoch(self, loader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        context = torch.enable_grad() if train else torch.no_grad()

        total_loss = correct = total = 0

        with context:
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                out  = self.model(imgs)
                loss = self.criterion(out, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(dim=1) == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, correct / total

    def train(self, save_path: str = 'best_model.pt', history_path: str = 'history.csv'):
        best_val_loss = float('inf')
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        header  = f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Status':>10}"
        divider = '-' * len(header)
        print(f"\nTraining for up to {self.epochs} epochs on {self.device}")
        print(divider)
        print(header)
        print(divider)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._train_epoch(self.val_loader,   train=False)

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            status = ''
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                status = 'saved best model'

            print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
                  f"{val_loss:>8.4f} | {val_acc:>8.2%} | {status:>10}")

        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"\nHistory saved to '{history_path}'")
        print(f"Best val loss : {best_val_loss:.4f}")