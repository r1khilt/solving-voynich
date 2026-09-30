"""Hardware-only recurrent source timing on random tokens; no corpus access."""
import hashlib
import json
import resource
import signal
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Probe(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(24, 96)
        self.recurrent = nn.LSTM(96, width, num_layers=2, batch_first=True)
        self.output = nn.Linear(width, 23)

    def forward(self, tokens):
        return self.output(self.recurrent(self.embedding(tokens))[0])


def main():
    def timeout(*_):
        raise TimeoutError('Hardware timing wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; do not start a language training run')
    rows = []
    for width in (512, 768):
        torch.manual_seed(90317)
        cpu_model = Probe(width).eval()
        small = torch.randint(0, 24, (2, 17))
        with torch.no_grad():
            expected = cpu_model(small)
        model = cpu_model.to('mps')
        with torch.no_grad():
            actual = model(small.to('mps')).cpu()
        delta = float((expected - actual).abs().max())
        if delta > 1e-4:
            raise ValueError('CPU/MPS forward check failed')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randint(0, 24, (16, 512), device='mps')
        y = torch.randint(0, 23, (16, 512), device='mps')
        timings = []
        for step in range(10):
            torch.mps.synchronize()
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(-1, 23), y.reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite random-token loss')
            loss.backward()
            optimizer.step()
            torch.mps.synchronize()
            elapsed = time.monotonic() - start
            if step >= 2:
                timings.append(elapsed)
        row = {'width': width, 'layers': 2, 'embedding': 96, 'batch': 16, 'length': 512,
               'parameters': sum(p.numel() for p in model.parameters()),
               'cpu_mps_maximum_logit_difference': delta, 'timed_step_seconds': timings,
               'mean_step_seconds': sum(timings) / len(timings),
               'mps_current_allocated_bytes': torch.mps.current_allocated_memory(),
               'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory()}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, cpu_model, optimizer, x, y
        torch.mps.empty_cache()
    result = {'status': 'random_token_hardware_probe_not_language_training', 'seed': 90317,
              'torch': torch.__version__, 'rows': rows, 'paid_cost_usd': 0,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection_rule': 'prefer768 if mean_step<=0.5s and driver_allocated<=8GiB, else512 under same limits, otherwise redesign budget'}
    target = Path(__file__).resolve().parents[1] / 'results/LATIN-SOURCE-001/hardware.json'
    with target.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
