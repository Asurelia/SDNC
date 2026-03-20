import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f'Tensor on GPU: {x}')
    print('OK')
else:
    print('NO GPU')
