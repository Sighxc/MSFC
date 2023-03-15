import torch

#对输入tensor进行量化
class tenbitquantz:
    def __init__(tensor):
        tensor = tensor
        scale = 0
        zero_point = 0
        quantized_tensor = 0
        dequantized_tensor = 0

    def quantize(tensor):
        reference = torch.tensor([15.0, -15.0], device='cuda')
        scale = (torch.max(reference) - torch.min(reference)) / (2**10 - 1)
        zero_point = torch.round(-torch.min(reference) / scale)
        quantized_tensor = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=torch.qint32)
        return quantized_tensor

    def dequantize(tensor):
        dequantized_tensor = torch.dequantize(tensor)
        return dequantized_tensor



