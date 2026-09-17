import numpy as np
import onnxruntime as ort


class ONNXInference:

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Input name :", self.input_name)
        print("Input shape:", self.session.get_inputs()[0].shape)
        print("Input type :", self.session.get_inputs()[0].type)

        print("Output name :", self.output_name)
        print("Output shape:", self.session.get_outputs()[0].shape)

    def predict(self, x):

        # PyTorch Tensor -> NumPy
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()

        x = np.asarray(
            x,
            dtype=np.float32
        )

        # Add batch dimension if needed
        if x.ndim == 3:
            x = np.expand_dims(
                x,
                axis=0
            )

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name: x
            }
        )[0]

        pred = np.argmax(
            logits,
            axis=1
        )

        return pred, logits