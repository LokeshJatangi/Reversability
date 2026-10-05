#!/usr/bin/env python3
"""Read-only audit of split Colab exports; write only a derived evidence JSON."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [part / 'artifacts' for part in args.parts]

    def resolve(relative):
        found = [root / relative for root in roots if (root / relative).is_file()]
        assert found, f'Missing export file: {relative}'
        if len(found) > 1:
            assert len({sha(p) for p in found}) == 1, f'Conflicting parts: {relative}'
        return found[0]

    def read(relative):
        return json.loads(resolve(relative).read_text())

    study = 'reversible-v1/'
    manifest_path = resolve(study + 'source-manifest.json')
    manifest_hash = sha(manifest_path)
    source_manifest = read(study + 'source-manifest.json')
    with zipfile.ZipFile(resolve(study + f'source-snapshots/{manifest_hash}.zip')) as archive:
        for name, digest in source_manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            assert sha(Path(name)) == digest, f'Current frozen source changed: {name}'

    data_root = 'data/data_fineweb_edu_gpt2_50m_v1/'
    data = read(data_root + 'manifest.json')
    data_hash = sha(resolve(data_root + 'manifest.json'))
    assets = {f'{name}.bin': record for name, record in data['files'].items()}
    assets.update({f'tokenizer/{name}': record for name, record in data['tokenizer']['files'].items()})
    for name, record in assets.items():
        path = resolve(data_root + name)
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name

    report = {'artifact_parts': [str(p) for p in args.parts],
              'source_manifest_sha256': manifest_hash, 'verified_source_files': len(source_manifest),
              'data_manifest_sha256': data_hash, 'verified_data_assets': assets,
              'runs': {}, 'checkpoint_audit': {}}
    names = [
        ('baseline', 'runs/baseline_20m_fineweb_edu_50m_v1', 'configs/baseline_colab.json', 50_000_000),
        ('baseline_smoke', 'runs/baseline_20m_fineweb_edu_50m_v1_smoke', 'configs/baseline_smoke_colab.json', 65_536),
        ('midpoint', study + 'runs/midpoint_20m_fineweb_edu_50m_v1_matched', study + 'configs/midpoint_matched.json', 50_000_000),
        ('euler', study + 'runs/euler_20m_fineweb_edu_50m_v1_matched', study + 'configs/euler_matched.json', 50_000_000),
        ('midpoint_smoke', study + 'runs/midpoint_20m_fineweb_edu_50m_v1_matched_smoke', study + 'configs/midpoint_smoke.json', 65_536),
        ('euler_smoke', study + 'runs/euler_20m_fineweb_edu_50m_v1_matched_smoke', study + 'configs/euler_smoke.json', 65_536),
        ('maximum', study + 'runs/midpoint_20m_fineweb_edu_50m_v1_maximum', study + 'configs/midpoint_maximum.json', 50_000_000),
    ]
    import torch
    for label, directory, config_path, budget in names:
        events = [json.loads(line) for line in resolve(directory + '/metrics.jsonl').read_text().splitlines()]
        starts = [e for e in events if e['event'] == 'startup']
        assert len(starts) == 1 and starts[0]['resumed_from_targets'] == 0, label
        startup = starts[0]
        summary = read(directory + '/run_summary.json')
        logged = [e for e in events if e['event'] == 'run_summary']
        assert len(logged) == 1 and {k: logged[0][k] for k in summary} == summary, label
        assert summary['committed_targets'] == summary['target_budget'] == budget
        assert summary['completed_full_target_budget']
        assert abs(summary['end_to_end_targets_per_second_this_process'] - budget / summary['elapsed_seconds_this_process']) < 1e-6
        config = read(config_path)
        assert sha(resolve(config_path)) == startup['config_sha256']
        assert startup['manifest_sha256'] == data_hash
        assert startup['parameters'] == 20_340_736
        train = [e for e in events if e['event'] == 'training_progress']
        assert all(a['committed_targets'] < b['committed_targets'] for a, b in zip(train, train[1:]))
        assert train[-1]['committed_targets'] == budget
        regular = config['physical_batch_size'] * config['gradient_accumulation_steps'] * 512
        assert summary['optimizer_steps'] == (budget + regular - 1) // regular
        assert train[-1]['valid_targets_this_update'] == budget - regular * (summary['optimizer_steps'] - 1)
        report['runs'][label] = {'summary': summary, 'startup': startup,
                                'event_counts': dict(Counter(e['event'] for e in events)),
                                'validation': [e for e in events if e['event'] == 'validation'],
                                'final_training': train[-1], 'config': config,
                                'config_sha256': sha(resolve(config_path))}
        for filename in ('latest.pt', 'best.pt'):
            relative = directory + '/' + filename
            if label.startswith('baseline'):
                # Checkpoint-free recovery export; earlier independent baseline audit covers these.
                report['checkpoint_audit'][relative] = {'status': 'absent in recovery export; audited in original baseline export'}
                continue
            path = resolve(relative)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['committed_targets'] == checkpoint['target_budget'] == budget
            assert checkpoint['step'] == summary['optimizer_steps']
            assert checkpoint['config_sha256'] == startup['config_sha256']
            assert checkpoint['manifest_sha256'] == data_hash
            assert checkpoint['last_validation_loss'] == summary['final_validation_loss']
            assert checkpoint['best_validation_loss'] == summary['best_validation_loss']
            for key in ('optimizer', 'grad_scaler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                assert key in checkpoint
            storages = {}
            for value in checkpoint['model'].values():
                assert torch.isfinite(value).all()
                storages[value.untyped_storage().data_ptr()] = value.numel()
            assert sum(storages.values()) == 20_340_736
            for state in checkpoint['optimizer']['state'].values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all()
            report['checkpoint_audit'][relative] = {'path': str(path), 'bytes': path.stat().st_size,
                                                  'sha256': sha(path), 'status': 'passed'}
            del checkpoint

    reference = report['runs']['baseline']
    controls = ('seed', 'sequence_length', 'model', 'precision', 'compile', 'eval_targets',
                'eval_interval', 'checkpoint_interval', 'learning_rate', 'min_learning_rate',
                'warmup_targets', 'weight_decay', 'beta1', 'beta2', 'grad_clip')
    for label in ('midpoint', 'euler', 'maximum'):
        record = report['runs'][label]
        for key in controls:
            assert record['config'][key] == reference['config'][key], (label, key)
        for key in ('torch', 'numpy', 'cuda_runtime', 'python', 'accelerator'):
            assert record['startup']['environment'][key] == reference['startup']['environment'][key], (label, key)
    for label in ('midpoint', 'euler'):
        for key in ('physical_batch_size', 'gradient_accumulation_steps'):
            assert report['runs'][label]['config'][key] == reference['config'][key]

    for name in ('review-decision.json', 'review-proposal.json', 'selection-policy.json',
                 'maximum-batch-accounting.json', 'benchmarks/midpoint_maximum_capacity.json',
                 'benchmarks/midpoint_maximum_confirmation.json'):
        report[name] = read(study + name)
    decision = report['review-decision.json']
    assert decision['planning_session_recorded'] and decision['selected_method'] == 'midpoint'
    assert decision['policy'] == report['selection-policy.json']
    assert decision['evidence']['source_manifest_sha256'] == manifest_hash
    assert decision['evidence']['review_proposal_sha256'] == sha(resolve(study + 'review-proposal.json'))
    assert decision['evidence']['cuda_gate_sha256'] == sha(resolve(study + 'correctness/cuda.json'))
    gate = read(study + 'correctness/cuda.json')
    assert gate['passed'] and len(gate['cases']) == 60
    for name, digest in gate['source_sha256'].items():
        relative = name.removeprefix('/content/Reversability/')
        assert source_manifest[relative] == digest
    report['cuda_gate'] = gate
    capacity = report['benchmarks/midpoint_maximum_capacity.json']
    assert capacity['selected_physical_batch_size'] == 33 and not capacity['search_was_capped']
    assert report['benchmarks/midpoint_maximum_confirmation.json']['status'] == 'pass'
    assert report['maximum-batch-accounting.json']['expected_optimizer_updates'] == 1480
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(names)} run streams, 10 reversible checkpoints, {len(source_manifest)} source hashes, data assets, decision and capacity; {args.output}')


if __name__ == '__main__':
    main()
