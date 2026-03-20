import torch, time

device = torch.device('cuda')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# Matrix mul
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    z = x @ y
torch.cuda.synchronize()
print(f'matmul 1000x1000 x100: {(time.time()-t0)*1000:.1f}ms')

# torch.outer (critical for Oja)
a = torch.randn(100, device=device)
b = torch.randn(200, device=device)
c = torch.outer(a, b)
print(f'torch.outer: {c.shape} — OK')

# torch.addcmul
t = torch.randn(100, device=device)
r = torch.addcmul(t, a, a, value=0.1)
print(f'torch.addcmul: OK')

# Softmax (router)
s = torch.randn(32, 1000, device=device)
w = torch.softmax(s, dim=-1)
print(f'softmax: OK')

# topk (router)
vals, idx = s.topk(5, dim=-1)
print(f'topk: OK')

# Batch operations (circuit bank pattern)
states = torch.zeros(1000, 9, device=device)
indices = torch.tensor([0, 5, 10, 15, 20], device=device)
selected = states[indices]
print(f'index_select: OK')

# CfC-like ODE step simulation
hidden = torch.randn(32, 64, device=device)
weight = torch.randn(64, 64, device=device)
bias = torch.randn(64, device=device)
out = torch.tanh(hidden @ weight + bias)
print(f'linear+tanh (CfC step): OK')

print('\nAll GPU ops passed!')
