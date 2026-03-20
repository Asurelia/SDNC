import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
    # Matrix mul test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = x @ y
    print(f'Matrix mul: {z.shape} on {z.device} — OK')
    # torch.outer test
    a = torch.randn(100, device='cuda')
    b = torch.randn(200, device='cuda')
    c = torch.outer(a, b)
    print(f'torch.outer: {c.shape} on {c.device} — OK')
    # addcmul test
    t = torch.randn(100, device='cuda')
    t1 = torch.randn(100, device='cuda')
    t2 = torch.randn(100, device='cuda')
    r = torch.addcmul(t, t1, t2, value=0.1)
    print(f'torch.addcmul: {r.shape} on {r.device} — OK')
else:
    print('NO GPU DETECTED')
