import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import ConfusionMatrixDisplay


class Plotter:
    def __init__(
        self,
        evaluator,
        log_history : list,
        model_name  : str  = "Model",
        output_dir  : str  = "./plots",
        dpi         : int  = 150,
    ):
        self.ev          = evaluator
        self.log_history = log_history
        self.model_name  = model_name
        self.output_dir  = output_dir
        self.dpi         = dpi
        os.makedirs(output_dir, exist_ok=True)


    def _parse_logs(self):
        train_loss_by_epoch = {}
        eval_entries        = []

        for entry in self.log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue

            if "eval_loss" in entry:
                # Eval log — keyed by integer epoch
                eval_entries.append(entry)
            elif "loss" in entry:
                # Step-level train loss — accumulate and average per epoch
                ep_int = int(epoch)
                if ep_int not in train_loss_by_epoch:
                    train_loss_by_epoch[ep_int] = []
                train_loss_by_epoch[ep_int].append(entry["loss"])

        # Average step-level train loss per epoch
        epochs     = sorted(train_loss_by_epoch.keys())
        train_loss = [np.mean(train_loss_by_epoch[e]) for e in epochs]

        # Align eval entries to integer epochs
        eval_entries  = sorted(eval_entries, key=lambda x: x["epoch"])
        eval_epochs   = [int(e["epoch"]) for e in eval_entries]
        val_loss      = [e.get("eval_loss",    np.nan) for e in eval_entries]
        val_acc       = [e.get("eval_val_acc", np.nan) for e in eval_entries]
        train_acc     = [e.get("train_acc",    np.nan) for e in eval_entries]

        return {
            "epochs"     : epochs,
            "train_loss" : train_loss,
            "eval_epochs": eval_epochs,
            "val_loss"   : val_loss,
            "val_acc"    : val_acc,
            "train_acc"  : train_acc,
        }

    def plot_confusion_matrix(self):
        ev           = self.ev
        label_names  = [ev.id2label[i] for i in sorted(ev.id2label)]

        fig, ax = plt.subplots(figsize=(5, 4))
        disp    = ConfusionMatrixDisplay(
            confusion_matrix = ev.confusion,
            display_labels   = label_names,
        )
        disp.plot(
            ax            = ax,
            cmap          = "Blues",
            colorbar      = False,
            values_format = "d",
        )
        ax.set_title(f"{self.model_name} — Confusion Matrix", fontsize=13, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)

        path = os.path.join(self.output_dir, f"{self.model_name.lower()}_confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        print(f"  Saved → {path}")

    def plot_roc_curve(self):
        ev      = self.ev
        auc_val = ev.metrics["auc_roc"]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            ev.fpr, ev.tpr,
            color     = "#2563EB",
            linewidth = 2,
            label     = f"AUC = {auc_val:.4f}",
        )
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1)
        ax.fill_between(ev.fpr, ev.tpr, alpha=0.08, color="#2563EB")

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"{self.model_name} — ROC Curve", fontsize=13, pad=10)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        path = os.path.join(self.output_dir, f"{self.model_name.lower()}_roc_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        print(f"  Saved → {path}")

    def plot_training_curves(self):
        logs = self._parse_logs()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{self.model_name} — Training Curves", fontsize=14, y=1.02)

        ax = axes[0]
        ax.plot(
            logs["epochs"],      logs["train_loss"],
            color="royalblue",   linewidth=2,  marker="o", markersize=4,
            label="Train Loss",
        )
        ax.plot(
            logs["eval_epochs"], logs["val_loss"],
            color="tomato",      linewidth=2,  marker="s", markersize=4,
            label="Val Loss",
        )
        ax.set_xlabel("Epoch",  fontsize=11)
        ax.set_ylabel("Loss",   fontsize=11)
        ax.set_title("Loss",    fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax = axes[1]
        ax.plot(
            logs["eval_epochs"], logs["train_acc"],
            color="royalblue",   linewidth=2,  marker="o", markersize=4,
            label="Train Acc",
        )
        ax.plot(
            logs["eval_epochs"], logs["val_acc"],
            color="tomato",      linewidth=2,  marker="s", markersize=4,
            label="Val Acc",
        )
        ax.set_xlabel("Epoch",    fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title("Accuracy",  fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        path = os.path.join(self.output_dir, f"{self.model_name.lower()}_training_curves.png")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    def plot_all(self):
        print(f"\n  Generating plots for {self.model_name} …")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_training_curves()
        print(f"  All plots saved to: {self.output_dir}")