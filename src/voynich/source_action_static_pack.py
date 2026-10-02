"""Fixed training geometry without changing the reading action features or loss.

This is a separate, prospectively qualified implementation. Existing registered
runs keep their original dynamic packer unchanged. It does not pad free decoding
or clear an allocator and makes no claim to solve a backend memory failure.
"""

import torch

from voynich.source_action_proposal import SourceActionProposal


class _CPUSchema:
    def __init__(self, config):
        self.config = config

    def parameters(self):
        yield torch.empty(0, device='cpu')


def static_pack(model, environments, traces, *, max_steps=448):
    """Validate on CPU, pad before the eight device transfers; no new features."""
    if type(max_steps) is not int or max_steps < 1 or max_steps > 896:
        raise ValueError('A bounded positive fixed training horizon is required')
    original = SourceActionProposal.pack(_CPUSchema(model.config), environments, traces)
    if original[1].shape[1] > max_steps:
        raise ValueError('Observed target trace exceeds fixed training horizon')
    packed = []
    fills = (model.config.glyphs, 0, 0, 0, 2*model.config.rows, False, False, 0)
    for index, (value, fill) in enumerate(zip(original, fills, strict=True)):
        shape = list(value.shape)
        shape[2 if index == 0 else 1] = model.config.max_glyphs if index == 0 else max_steps
        padded = value.new_full(shape, fill)
        if index == 5:
            padded[..., 0] = True  # Same harmless legal dummy for inactive queries.
        padded[tuple(slice(0, n) for n in value.shape)] = value
        packed.append(padded.to(next(model.parameters()).device))
    return tuple(packed)
