"""Hebbian learning: local weight updates without global backpropagation.

Implements Oja's rule (stable Hebbian) and synaptic pruning.
"Circuits that fire together wire together."

All updates are LOCAL — only active circuits are modified.

The CfC internal structure has 3 layers:
  Layer 0 (sensory→inter):  ff1.weight(inter, input_dim + inter)
  Layer 1 (inter→command):  ff1.weight(command, inter + command)
  Layer 2 (command→motor):  ff1.weight(motor, command + motor)

Each layer also has a sparsity_mask (same shape as ff1.weight) that we
must respect — only update where the mask is nonzero.
"""

import torch


def oja_update(
    weight: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    lr: float = 1e-4,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Oja's learning rule — stable Hebbian update.

    w_new = w + lr * post * (pre - post * w)

    Uses unsqueeze-based outer product for ROCm compatibility
    (avoids potential torch.outer issues on some ROCm versions).

    Args:
        weight: (out_features, in_features)
        pre: pre-synaptic activations (batch, in_features)
        post: post-synaptic activations (batch, out_features)
        lr: learning rate
        mask: optional sparsity mask — only update where mask != 0

    Returns:
        Updated weight tensor.
    """
    pre_mean = pre.mean(0)    # (in_features,)
    post_mean = post.mean(0)  # (out_features,)

    # Oja: dw = lr * y * (x - y^T @ W)
    # Using unsqueeze for ROCm-safe outer product
    residual = pre_mean - post_mean @ weight  # (in_features,)
    delta = lr * post_mean.unsqueeze(1) * residual.unsqueeze(0)  # (out, in)

    if mask is not None:
        delta = delta * (mask != 0).float()

    return weight + delta


@torch.no_grad()
def apply_hebbian_to_cfc(
    cfc_module: torch.nn.Module,
    input_tensor: torch.Tensor,
    hidden_state: torch.Tensor,
    lr: float = 1e-4,
):
    """Apply Oja update layer-by-layer inside a CfC cell.

    Hooks into the CfC's internal 3-layer NCP structure:
      Layer 0: sensory → inter (input is [input, inter_state])
      Layer 1: inter → command (input is [inter_output, command_state])
      Layer 2: command → motor (input is [command_output, motor_state])

    Respects sparsity masks — only updates wired connections.

    Args:
        cfc_module: The CfC module (MicroCircuit.cfc).
        input_tensor: (batch, input_dim) — raw input to the circuit.
        hidden_state: (batch, state_size) — hidden state AFTER forward.
        lr: Hebbian learning rate.
    """
    # The CfC rnn_cell has layer_0, layer_1, layer_2
    rnn_cell = None
    for name, module in cfc_module.named_modules():
        if name == 'rnn_cell':
            rnn_cell = module
            break

    if rnn_cell is None:
        return

    # Parse hidden state into neuron groups
    # NCP wiring: inter=4, command=3, motor=2 (from config)
    # hidden_state is (batch, state_size=9) = [inter(4), command(3), motor(2)]
    layers = []
    for i in range(10):
        layer_name = f'layer_{i}'
        if hasattr(rnn_cell, layer_name):
            layers.append(getattr(rnn_cell, layer_name))
        else:
            break

    if len(layers) < 1:
        return

    # Align batch sizes: input and hidden may differ (expert choice routing)
    # Use mean to get representative activations for Hebbian update
    if input_tensor.shape[0] != hidden_state.shape[0]:
        # Expand the smaller one to a single sample via mean
        inp = input_tensor.mean(0, keepdim=True)
        hid = hidden_state.mean(0, keepdim=True)
    else:
        inp = input_tensor
        hid = hidden_state

    state_offset = 0
    prev_activation = inp

    for layer in layers:
        if not hasattr(layer, 'ff1'):
            continue

        w = layer.ff1.weight  # (out_neurons, in_features)
        out_neurons = w.shape[0]
        in_features = w.shape[1]

        # Post-synaptic: slice of hidden state for this layer's neurons
        post = hid[:, state_offset:state_offset + out_neurons]

        # Pre-synaptic: prev_activation padded to match in_features
        pre_dim = prev_activation.shape[-1]
        if pre_dim >= in_features:
            pre = prev_activation[:, :in_features]
        else:
            need = in_features - pre_dim
            state_slice = hid[:, state_offset:state_offset + need]
            actual = state_slice.shape[1]
            if actual < need:
                padding = torch.zeros(
                    prev_activation.shape[0], need - actual,
                    device=prev_activation.device,
                )
                pre = torch.cat([prev_activation, state_slice, padding], dim=-1)
            else:
                pre = torch.cat([prev_activation, state_slice], dim=-1)

        mask = None
        if hasattr(layer, 'sparsity_mask'):
            mask = layer.sparsity_mask.data

        layer.ff1.weight.data = oja_update(w.data, pre, post, lr, mask)

        prev_activation = post
        state_offset += out_neurons


@torch.no_grad()
def apply_hebbian_to_circuit(
    circuit_module: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    lr: float = 1e-4,
):
    """Apply Oja update to a module's linear layers (for output_proj etc).

    Simple version — matches dimensions directly.
    """
    in_dim = input_tensor.shape[-1]
    out_dim = output_tensor.shape[-1]

    pre = input_tensor.reshape(-1, in_dim) if input_tensor.dim() > 2 else input_tensor
    post = output_tensor.reshape(-1, out_dim) if output_tensor.dim() > 2 else output_tensor

    for param in circuit_module.parameters():
        if param.dim() != 2:
            continue

        p_out, p_in = param.shape

        if p_in == in_dim and p_out == out_dim:
            param.data = oja_update(param.data, pre, post, lr)
        elif p_in == in_dim and p_out <= out_dim:
            param.data = oja_update(param.data, pre, post[:, :p_out], lr)
        elif p_in <= in_dim and p_out == out_dim:
            param.data = oja_update(param.data, pre[:, :p_in], post, lr)


@torch.no_grad()
def synaptic_pruning(module: torch.nn.Module, threshold: float = 1e-5):
    """Prune weak synaptic connections.

    Respects sparsity_mask — never prunes masked-out connections
    (they're already zero by design).
    """
    sparsity_masks = {}
    for name, param in module.named_parameters():
        if 'sparsity_mask' in name:
            # Map to the corresponding ff1.weight
            key = name.replace('sparsity_mask', 'ff1.weight')
            sparsity_masks[key] = param.data

    for name, param in module.named_parameters():
        if param.dim() >= 2 and 'sparsity_mask' not in name:
            mask = param.data.abs() > threshold
            # Don't prune where sparsity_mask already controls structure
            if name in sparsity_masks:
                structural_mask = sparsity_masks[name] != 0
                mask = mask | (~structural_mask)  # Keep structurally-zero as-is
            param.data *= mask.float()


@torch.no_grad()
def strengthen_co_active_circuits(
    co_activation_matrix: torch.Tensor,
    connection_weights: torch.Tensor,
    lr: float = 1e-4,
    decay: float = 0.999,
) -> torch.Tensor:
    """Strengthen connections between co-active circuits (Hebbian inter-circuit)."""
    total = co_activation_matrix.sum()
    if total > 0:
        normalized = co_activation_matrix / total
        connection_weights = decay * connection_weights + lr * normalized
    return connection_weights
