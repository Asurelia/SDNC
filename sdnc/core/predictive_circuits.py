"""Predictive Circuits — each circuit generates hypotheses and tests them.

Inspired by Rao & Ballard 1999, Karl Friston Free Energy Principle.

Each circuit has TWO heads:
  - Prediction head: "what do I expect next?"
  - Processing head: "what did I actually get?"

The prediction error (actual - predicted) is THE learning signal.
No external loss needed — the brain learns by minimizing surprise.
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import NCP

from sdnc.config import SDNCConfig


def build_prediction_wiring(config: SDNCConfig) -> NCP:
    """Build NCP wiring for prediction head — smaller than processing."""
    return NCP(
        inter_neurons=config.ncp_inter_neurons // 2,
        command_neurons=config.ncp_command_neurons // 2,
        motor_neurons=config.input_dim,  # predicts in input space
        sensory_fanout=config.ncp_sensory_fanout,
        inter_fanout=config.ncp_inter_fanout,
        recurrent_command_synapses=config.ncp_recurrent_command_synapses,
        motor_fanin=config.ncp_motor_fanin,
    )


class PredictiveCircuit(nn.Module):
    """A single circuit with dual prediction/processing heads.

    Processing head: standard CfC — transforms input into circuit output.
    Prediction head: CfC that predicts the NEXT input from current state.

    The prediction error drives Hebbian learning locally.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config

        # Processing head — same as original MicroCircuit
        proc_wiring = NCP(
            inter_neurons=config.ncp_inter_neurons,
            command_neurons=config.ncp_command_neurons,
            motor_neurons=config.circuit_output_dim,
            sensory_fanout=config.ncp_sensory_fanout,
            inter_fanout=config.ncp_inter_fanout,
            recurrent_command_synapses=config.ncp_recurrent_command_synapses,
            motor_fanin=config.ncp_motor_fanin,
        )
        self.processing_cfc = CfC(
            input_size=config.input_dim,
            units=proc_wiring,
            batch_first=True,
            return_sequences=False,
            mode="pure",
        )
        self._proc_state_size = proc_wiring.units

        # Prediction head — predicts next input from current hidden state
        # Input: current hidden state → Output: predicted next input (input_dim)
        pred_wiring = build_prediction_wiring(config)
        self.prediction_cfc = CfC(
            input_size=self._proc_state_size,  # reads from processing state
            units=pred_wiring,
            batch_first=True,
            return_sequences=False,
            mode="pure",
        )
        self._pred_state_size = pred_wiring.units

    @property
    def proc_state_size(self) -> int:
        return self._proc_state_size

    @property
    def pred_state_size(self) -> int:
        return self._pred_state_size

    @property
    def total_state_size(self) -> int:
        return self._proc_state_size + self._pred_state_size

    @property
    def output_size(self) -> int:
        return self.processing_cfc.output_size

    def forward(
        self,
        x: torch.Tensor,
        proc_state: torch.Tensor | None = None,
        pred_state: torch.Tensor | None = None,
    ) -> dict:
        """Forward with prediction.

        Args:
            x: Input (batch, input_dim) — the ACTUAL input received.
            proc_state: Processing CfC hidden state.
            pred_state: Prediction CfC hidden state.

        Returns:
            dict with:
                output: circuit processing output (batch, circuit_output_dim)
                prediction: predicted input (batch, input_dim)
                prediction_error: actual - predicted (batch, input_dim)
                new_proc_state: updated processing state
                new_pred_state: updated prediction state
        """
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)  # (B, 1, D)
        else:
            x_seq = x

        # 1. Generate prediction BEFORE seeing input (from current state)
        if proc_state is not None:
            pred_input = proc_state.unsqueeze(1)  # (B, 1, proc_state_size)
            prediction, new_pred_state = self.prediction_cfc(
                pred_input, hx=pred_state
            )
        else:
            # First step: no state yet → prediction is zeros
            batch_size = x.shape[0]
            prediction = torch.zeros(
                batch_size, self.config.input_dim, device=x.device
            )
            new_pred_state = pred_state

        # 2. Compute prediction error
        x_flat = x if x.dim() == 2 else x[:, -1, :]
        prediction_error = x_flat - prediction

        # 3. Process actual input through processing head
        output, new_proc_state = self.processing_cfc(x_seq, hx=proc_state)

        return {
            "output": output,
            "prediction": prediction,
            "prediction_error": prediction_error,
            "new_proc_state": new_proc_state,
            "new_pred_state": new_pred_state,
        }


class PredictiveCircuitBank(nn.Module):
    """Bank of N predictive circuits — shared weights, independent states.

    Each circuit maintains TWO states:
      - Processing state: evolves with input processing
      - Prediction state: evolves with prediction generation

    Prediction errors are computed per-circuit and aggregated.
    """

    def __init__(self, config: SDNCConfig):
        super().__init__()
        self.config = config
        self.n_circuits = config.n_circuits

        # Shared circuit (one set of weights, N states)
        self.circuit = PredictiveCircuit(config)

        # Independent states per circuit
        self.register_buffer(
            "proc_states",
            torch.zeros(config.n_circuits, self.circuit.proc_state_size),
        )
        self.register_buffer(
            "pred_states",
            torch.zeros(config.n_circuits, self.circuit.pred_state_size),
        )
        self.register_buffer(
            "activation_history", torch.zeros(config.n_circuits)
        )
        self.register_buffer(
            "prediction_error_history",
            torch.zeros(config.n_circuits),
        )

        # Inter-circuit wiring (preserved from original)
        self.register_buffer(
            "inter_circuit_weights",
            torch.zeros(config.n_circuits, config.n_circuits),
        )

    @property
    def output_size(self) -> int:
        return self.circuit.output_size

    def forward_token_choice(
        self,
        x: torch.Tensor,
        active_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> dict:
        """Token-choice routing with prediction.

        Returns dict with circuit outputs, predictions, and errors.
        """
        batch_size = x.shape[0]
        k = active_indices.shape[1]
        output_dim = self.output_size
        input_dim = self.config.input_dim

        all_outputs = torch.zeros(batch_size, k, output_dim, device=x.device)
        all_predictions = torch.zeros(batch_size, k, input_dim, device=x.device)
        all_errors = torch.zeros(batch_size, k, input_dim, device=x.device)
        last_proc_state = None

        for i in range(k):
            slot_indices = active_indices[:, i]

            # Get both states for selected circuits
            sel_proc = self.proc_states[slot_indices]
            sel_pred = self.pred_states[slot_indices]

            result = self.circuit(x, proc_state=sel_proc, pred_state=sel_pred)

            # Update states (no grad — liquid dynamics)
            self.proc_states[slot_indices] = result["new_proc_state"].detach()
            if result["new_pred_state"] is not None:
                self.pred_states[slot_indices] = result["new_pred_state"].detach()

            all_outputs[:, i, :] = result["output"]
            all_predictions[:, i, :] = result["prediction"]
            all_errors[:, i, :] = result["prediction_error"]
            last_proc_state = result["new_proc_state"]

            # Track activations and prediction errors
            with torch.no_grad():
                for idx in slot_indices:
                    self.activation_history[idx] += 1
                error_magnitude = result["prediction_error"].norm(dim=-1).mean()
                for idx in slot_indices:
                    # EMA of prediction error per circuit
                    self.prediction_error_history[idx] = (
                        0.95 * self.prediction_error_history[idx]
                        + 0.05 * error_magnitude
                    )

        # Weighted aggregation
        weighted_output = (all_outputs * weights.unsqueeze(-1)).sum(dim=1)
        weighted_prediction = (all_predictions * weights.unsqueeze(-1)).sum(dim=1)
        weighted_error = (all_errors * weights.unsqueeze(-1)).sum(dim=1)

        return {
            "circuit_output": weighted_output,
            "prediction": weighted_prediction,
            "prediction_error": weighted_error,
            "hidden_states": last_proc_state,
            "per_circuit_errors": all_errors,
        }

    def forward_expert_choice(
        self,
        x: torch.Tensor,
        routing_info: tuple,
        top_weights: torch.Tensor,
    ) -> dict:
        """Expert-choice routing with prediction."""
        assignments, weights, circuit_map, counts_per_circuit = routing_info
        batch_size = x.shape[0]
        output_dim = self.output_size
        input_dim = self.config.input_dim
        device = x.device

        token_outputs = torch.zeros(batch_size, output_dim, device=device)
        token_predictions = torch.zeros(batch_size, input_dim, device=device)
        token_errors = torch.zeros(batch_size, input_dim, device=device)
        token_weight_sums = torch.zeros(batch_size, 1, device=device)
        last_proc_state = None

        n_active = circuit_map.shape[0]

        for i in range(n_active):
            cid = circuit_map[i].item()
            n_tokens = counts_per_circuit[i].item()
            if n_tokens == 0:
                continue

            token_ids = assignments[i, :n_tokens]
            w = weights[i, :n_tokens]

            circuit_inputs = x[token_ids]
            proc_state = self.proc_states[cid].unsqueeze(0).expand(n_tokens, -1)
            pred_state = self.pred_states[cid].unsqueeze(0).expand(n_tokens, -1)

            result = self.circuit(
                circuit_inputs, proc_state=proc_state, pred_state=pred_state
            )

            # Update states (mean across tokens for this circuit)
            self.proc_states[cid] = result["new_proc_state"].detach().mean(dim=0)
            if result["new_pred_state"] is not None:
                self.pred_states[cid] = result["new_pred_state"].detach().mean(dim=0)
            last_proc_state = result["new_proc_state"]

            weighted_out = result["output"] * w.unsqueeze(-1)
            weighted_pred = result["prediction"] * w.unsqueeze(-1)
            weighted_err = result["prediction_error"] * w.unsqueeze(-1)

            for j in range(n_tokens):
                tid = token_ids[j]
                token_outputs[tid] += weighted_out[j]
                token_predictions[tid] += weighted_pred[j]
                token_errors[tid] += weighted_err[j]
                token_weight_sums[tid] += w[j]

            with torch.no_grad():
                self.activation_history[cid] += n_tokens
                error_mag = result["prediction_error"].norm(dim=-1).mean()
                self.prediction_error_history[cid] = (
                    0.95 * self.prediction_error_history[cid] + 0.05 * error_mag
                )

        safe_sums = token_weight_sums.clamp(min=1e-8)
        token_outputs = token_outputs / safe_sums
        token_predictions = token_predictions / safe_sums
        token_errors = token_errors / safe_sums

        return {
            "circuit_output": token_outputs,
            "prediction": token_predictions,
            "prediction_error": token_errors,
            "hidden_states": last_proc_state,
            "per_circuit_errors": None,
        }

    def get_states(self, indices: torch.Tensor) -> torch.Tensor:
        return self.proc_states[indices]

    def reset_states(self):
        """Reset all states — NEVER call this during training stream."""
        self.proc_states.zero_()
        self.pred_states.zero_()

    def decay_states(self, rate: float = 0.99):
        self.proc_states *= rate
        self.pred_states *= rate
