import timm
import torch
import numpy as np

class Inference:
    def __init__(
        self,
        model_path: str,
        checkpoint_path: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            "convnext_tiny_hnf.a2h_in1k",
            pretrained=False,
            in_chans=1,
            num_classes=2,
        ).to(self.device)

        state_dict = torch.load(checkpoint_path)

        self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, x):
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, self.predict].item()

        return pred, conf   