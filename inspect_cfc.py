import torch
from sdnc.config import SDNCConfig
from sdnc.core.micro_circuit import MicroCircuit

config = SDNCConfig(input_dim=512, circuit_dim=8, circuit_output_dim=2)
mc = MicroCircuit(config)

print("=== CfC internal parameter shapes ===")
for name, param in mc.cfc.named_parameters():
    print(f"  {name:40s} {str(list(param.shape)):20s} numel={param.numel()}")

print(f"\nstate_size = {mc.state_size}")
print(f"output_size = {mc.output_size}")
print(f"input_dim = {config.input_dim}")

# What we need for Oja: identify (in_dim, out_dim) pairs
print("\n=== Candidate layers for Hebbian ===")
for name, param in mc.cfc.named_parameters():
    if param.dim() == 2:
        out_d, in_d = param.shape
        print(f"  {name}: weight({out_d}, {in_d})")
        print(f"    pre={in_d}, post={out_d}")
