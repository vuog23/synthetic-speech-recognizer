import numpy as np
from onnxruntime.quantization import CalibrationDataReader

class CNNCalibrationDataReader(CalibrationDataReader):

    def __init__(
        self,
        dataloader,
        input_name,
        max_samples=500,
    ):
        self.dataloader = dataloader
        self.input_name = input_name
        self.max_samples = max_samples

        self.iterator = None
        self.pending_samples = []
        self.sample_count = 0

        self.rewind()

    def _extract_input(self, batch):
        return batch[0]

    def get_next(self):

        if self.sample_count >= self.max_samples:
            return None

        while len(self.pending_samples) == 0:

            try:
                batch = next(self.iterator)

            except StopIteration:
                return None

            x = (
                self._extract_input(batch)
                .detach()
                .cpu()
                .float()
            )

            for i in range(x.shape[0]):
                self.pending_samples.append(
                    x[i:i + 1]
                )

        sample = self.pending_samples.pop(0)

        self.sample_count += 1

        return {
            self.input_name:
                sample.numpy().astype(
                    np.float32
                )
        }

    def rewind(self):

        self.iterator = iter(
            self.dataloader
        )

        self.pending_samples = []
        self.sample_count = 0