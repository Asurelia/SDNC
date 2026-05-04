# SDNC Research Notes

Last web check: 2026-05-02.

## What Is Still Viable

- Liquid Time-Constant Networks remain relevant for dynamic recurrent systems:
  https://arxiv.org/abs/2006.04439
- CfC models are a faster closed-form continuation of the liquid-network line:
  https://arxiv.org/abs/2106.13898
- Neural Circuit Policies remain the right reference for compact sparse
  biologically inspired circuit wiring:
  https://www.nature.com/articles/s42256-020-00237-3
- The `ncps` package is still available for PyTorch/TensorFlow/Keras:
  https://pypi.org/project/ncps/

## What Evolved

- Liquid AI now presents Liquid Foundation Models as efficient hybrid models for
  on-device/cloud deployment:
  https://www.liquid.ai/models
- The adult Drosophila connectome is now mapped at whole-brain scale, with
  around 140k neurons and more than 50M synapses:
  https://www.nih.gov/news-events/nih-research-matters/complete-wiring-map-adult-fruit-fly-brain
  https://www.nature.com/articles/s41586-024-07558-y
- Test-time/adaptive memory is active research again:
  https://arxiv.org/abs/2501.00663
  https://arxiv.org/abs/2402.04624
- Agent learning from feedback and memory is well explored with LLM agents, but
  usually the LLM remains the central brain:
  https://arxiv.org/abs/2303.11366
- Hugging Face Datasets supports streaming large datasets as iterable datasets:
  https://huggingface.co/docs/datasets/v2.14.5/en/stream
- Hugging Face Datasets has typed features for image and audio columns:
  https://huggingface.co/docs/datasets/about_dataset_features

## SDNC Position

The implementation follows the useful pieces without adopting the heavy pieces:

- From liquid/NCP work: sparse, stateful, dynamic circuits.
- From connectomics: modular specialized circuits and sparse wiring.
- From Reflexion-like agents: feedback and episodic memory.
- From test-time memory work: updating memory while deployed.
- From dataset tooling: stream data as experiences instead of materializing an
  enormous dense training corpus in memory.

The difference is that SDNC puts the lightweight local learner at the center.
LLMs, if added later, must stay as optional peripheral tools.
