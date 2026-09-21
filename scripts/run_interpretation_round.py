"""Execute only the fixed EXP-0003/0004 registrations; no paid API or manuscript test scoring."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    selected = json.loads(Path('results/EXP-0002/comparison.json').read_text())['selected_for_interpretation']
    commands = []
    for seed in [42, 43, 44]:
        commands.append([sys.executable, '-m', 'voynich.context_analysis', '--checkpoint',
                         f'outputs/EXP-0002/{selected}-seed{seed}/best.pt', '--device', 'mps', '--output',
                         f'outputs/EXP-0003/{selected}-seed{seed}.json', *(['--heads'] if seed == 42 else [])])
    for family in ['cycle_null', 'iid']:
        for seed in [42, 43, 44]:
            root = f'outputs/EXP-0004/{family}-seed{seed}'
            commands.append([sys.executable, '-m', 'voynich.train', '--config', 'configs/synthetic_small.json',
                             '--data', f'data/processed/synthetic_{family}', '--run-dir', root,
                             '--seed', str(seed), '--device', 'mps'])
            commands.append([sys.executable, '-m', 'voynich.calibration', '--checkpoint', root+'/best.pt',
                             '--data', f'data/processed/synthetic_{family}', '--family', family,
                             '--device', 'mps', '--output', root+'/calibration.json',
                             *(['--causal'] if seed == 42 and family == 'cycle_null' else [])])
    print(json.dumps({'commands': commands, 'execute': args.execute, 'manuscript_test_scored': False}, indent=2), flush=True)
    if args.execute:
        for command in commands:
            print('Running: '+' '.join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
