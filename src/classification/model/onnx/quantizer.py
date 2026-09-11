from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def quantize_int8(
    fp32_path,
    int8_path,
    calibration_reader,
):

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,

        quant_format=QuantFormat.QDQ,

        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,

        per_channel=True,

        calibrate_method=CalibrationMethod.MinMax,

        op_types_to_quantize=[
            "Conv",
            "MatMul",
            "Gemm",
        ],
    )