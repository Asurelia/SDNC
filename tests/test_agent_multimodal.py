import numpy as np

from sdnc.agent.multimodal import LocalMultimodalEncoder, ModalitySample, samples_from_record


def test_multimodal_encoder_handles_text_image_audio_and_video():
    encoder = LocalMultimodalEncoder(dim=64)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    audio = {"array": np.sin(np.linspace(0, 3.14, 160)).astype(np.float32), "sampling_rate": 16000}
    video = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8) * 255]
    samples = [
        ModalitySample("text", text="corrige ce bug python", label="code"),
        ModalitySample("image", content=image, text="image rouge", label="red"),
        ModalitySample("audio", content=audio, text="signal court", label="tone"),
        ModalitySample("video", content=video, text="transition", label="motion"),
    ]

    encoded = [encoder.encode_sample(sample) for sample in samples]

    assert all(item.embedding.shape == (64,) for item in encoded)
    assert all(np.isclose(np.linalg.norm(item.embedding), 1.0) for item in encoded)
    assert encoded[1].features["available"] is True
    assert encoded[2].features["sampling_rate"] == 16000
    assert encoded[3].features["motion"] > 0


def test_samples_from_record_infers_modalities():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    record = {"caption": "un carré noir", "image": image, "label": "shape"}

    samples = samples_from_record(record, source="fixture")

    assert {sample.modality for sample in samples} == {"image", "text"}
    assert all(sample.label == "shape" for sample in samples)
