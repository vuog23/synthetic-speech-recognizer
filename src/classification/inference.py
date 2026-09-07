import timm
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test = torch.load(r"D:\Project\Synthetic Speech Recognizer\datasets\processed\test.pt", weights_only=False)

model = timm.create_model(
    "convnext_tiny_hnf.a2h_in1k",
    pretrained=False,
    in_chans=1,
    num_classes=2,
).to(device)

state_dict = torch.load(
    r"D:\Project\Synthetic Speech Recognizer\models\convnext_tiny\best_model.pt",
)

model.load_state_dict(state_dict)

model.eval()

idx = np.random.randint(len(test['label']))

test_sample = test['data'][idx]
test_label = test['label'][idx]

x = test_sample.unsqueeze(0).to(device)

with torch.inference_mode():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, prediction].item()

print(f"Ground truth: {test_label.item()}")
print(f"Predicted: {prediction}")
print(f"Confidence: {confidence:.4f}")