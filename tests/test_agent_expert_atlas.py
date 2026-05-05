import numpy as np

from sdnc.agent.expert_atlas import (
    build_codebook_payload,
    build_low_rank_payload,
    build_procedure_payload,
    build_prototype_payload,
    build_sparse_delta_payload,
    decode_payload,
    payload_summary,
)


def test_procedure_payload_decodes_by_level_with_stable_checksum():
    embedding = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    first = build_procedure_payload(
        name="expert:calculator:addition",
        description="Use calculator for arithmetic.",
        tool_name="calculator",
        trigger_embedding=embedding,
        payload={"claim": "calculator handles addition"},
    )
    second = build_procedure_payload(
        name="expert:calculator:addition",
        description="Use calculator for arithmetic.",
        tool_name="calculator",
        trigger_embedding=embedding,
        payload={"claim": "calculator handles addition"},
    )

    assert first.checksum == second.checksum
    assert first.byte_size > 0
    assert first.hot_vram_gb == 0.0

    l0 = decode_payload(first, level="L0")
    l1 = decode_payload(first.to_dict(), level="L1")
    l2 = decode_payload(first.to_dict(), level="L2")

    assert l0.integrity_ok
    assert l0.data["tool_name"] == "calculator"
    assert "trigger_sketch" not in l0.data
    assert l1.data["trigger_sketch"]["dim"] == 16
    assert l2.data["evidence"]["claim"] == "calculator handles addition"


def test_payload_tamper_breaks_integrity():
    payload = build_prototype_payload("pattern", np.ones(8, dtype=np.float32))
    stored = payload.to_dict()
    stored["data"]["levels"]["L0"]["threshold"] = 0.1

    decoded = decode_payload(stored, level="L0")

    assert not decoded.integrity_ok


def test_low_rank_sparse_delta_and_codebook_payloads_decode():
    low_rank = build_low_rank_payload(
        "rank-two",
        left=np.eye(3, 2, dtype=np.float32),
        right=np.ones((2, 4), dtype=np.float32),
        bias=np.zeros(4, dtype=np.float32),
    )
    sparse_delta = build_sparse_delta_payload(
        "delta",
        shape=(4, 4),
        indices=[[0, 1], [3, 2]],
        values=[0.25, -0.5],
    )
    codebook = build_codebook_payload(
        "codes",
        codebook=[[0.0, 1.0], [1.0, 0.0]],
        codes=[0, 1, 1, 0],
        output_shape=(2, 4),
    )

    assert decode_payload(low_rank, level="L0").data["rank"] == 2
    assert decode_payload(low_rank, level="L2").data["right"][0] == [1.0, 1.0, 1.0, 1.0]
    assert decode_payload(sparse_delta, level="L1").data["nnz"] == 2
    assert decode_payload(codebook, level="L2").data["codes"] == [0, 1, 1, 0]
    assert payload_summary(codebook)["integrity_ok"]
