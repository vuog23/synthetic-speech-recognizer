from src.classification.training.trainer import Trainer
from src.classification.evaluation.evaluator import Evaluator

data_root = (
    r"D:\Project\Synthetic Speech Recognizer"
    r"\datasets\ASVLibri"
)

trainer = Trainer(
    model_path="timm/convnextv2_base.fcmae_ft_in1k",

    data_root=data_root,

    in_chans=1,
    num_classes=2,

    batch_size=32,
    lr=1e-4,
    drop_path_rate=0.0,
    max_grad_norm=1.0,
    use_class_weights=True,
    check_finite=True,

    epochs=10,
)

trainer.train(
    save_path="best_model.pt",
    history_path="history.csv",
)

evaluator = Evaluator(
    model_path="timm/convnextv2_base.fcmae_ft_in1k",
    weight_path="best_model.pt",

    in_chans=1,
    num_classes=2,

    batch_size=32,
)

results = evaluator.evaluate(
    data_root=data_root
)

print(results)
