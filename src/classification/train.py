from training.trainer import Trainer
from evaluation.evaluator import Evaluator

data_root = (
    r"D:\Project\Synthetic Speech Recognizer"
    r"\datasets\raw\for-norm\for-norm"
)

trainer = Trainer(
    model_path="convnext_tiny_hnf.a2h_in1k",

    data_root=data_root,

    in_chans=1,
    num_classes=2,

    batch_size=32,
    lr=1e-3,
    weight_decay=1e-5,

    epochs=10,
)

trainer.train(
    save_path="best_model.pt",
    history_path="history.csv",
)

evaluator = Evaluator(
    model_path="convnext_tiny_hnf.a2h_in1k",
    weight_path="best_model.pt",

    in_chans=1,
    num_classes=2,

    batch_size=32,
)

results = evaluator.evaluate(
    data_root=data_root
)

print(results)