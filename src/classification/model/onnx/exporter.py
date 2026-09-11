import torch


def export_onnx(
    model,
    example_input,
    save_path,
):
    model = model.eval().cpu()
    example_input = example_input.cpu()

    torch.onnx.export(
        model,
        (example_input,),
        save_path,
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
    )