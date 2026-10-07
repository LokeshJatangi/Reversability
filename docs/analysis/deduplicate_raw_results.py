#!/usr/bin/env python3
"""Plan, then remove only SHA-256-identical untracked raw result copies."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rank(record):
    name = record['original_path']
    if name.startswith('artifacts-20261004T124446Z-1-001/'):
        return (0, name)
    if name.startswith('artifacts-20261004T124446Z-1-002/'):
        return (1, name)
    if name.startswith('artifacts-20261004T124446Z'):
        return (2, name)
    return (3, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    inventory = json.loads(args.inventory.read_text())
    groups = defaultdict(list)
    for record in inventory['raw_files']:
        groups[record['sha256']].append(record)
    retained, removed = [], []
    for digest, records in sorted(groups.items()):
        ordered = sorted(records, key=rank)
        canonical = ordered[0]
        retained.append(canonical)
        for record in ordered[1:]:
            removed.append({**record, 'retained_path': canonical['original_path']})
    expected = {
        'format_version': 1,
        'inventory_sha256': sha(args.inventory),
        'policy': 'Remove only byte-identical untracked raw copies; preserve each unique SHA-256 and prefer final split exports. Resolve old paths through retained_path.',
        'retained_files': retained,
        'duplicate_files': removed,
        'retained_unique_files': len(retained),
        'duplicate_file_count': len(removed),
        'redundant_bytes': sum(r['bytes'] for r in removed),
    }
    if not args.apply:
        if args.plan.exists():
            raise RuntimeError('Plan already exists; preserve it instead of overwriting')
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps({**expected, 'status': 'planned'}, indent=2) + '\n')
        print(f'PLANNED: retain {len(retained)} unique raw files; remove {len(removed)} duplicates / {expected["redundant_bytes"]:,} bytes')
        return

    plan = json.loads(args.plan.read_text())
    if {k: plan[k] for k in expected} != expected or plan['status'] != 'planned':
        raise RuntimeError('Plan changed, inventory changed, or cleanup already completed')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0'))
    roots = [Path(target) for target in inventory['raw_targets']]

    def checked_path(name):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.is_symlink():
            raise RuntimeError(f'Unsafe path: {name}')
        if not path.resolve().is_relative_to(root) or name in tracked:
            raise RuntimeError(f'Outside workspace or tracked: {name}')
        if not any(path == target or path.is_relative_to(target) for target in roots):
            raise RuntimeError(f'Outside inventoried raw targets: {name}')
        if not path.is_file():
            raise RuntimeError(f'Missing raw file: {name}')
        return path

    # Verify every surviving unique file and every deletion before deleting anything.
    for record in [*retained, *removed]:
        path = checked_path(record['original_path'])
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise RuntimeError(f'Raw evidence changed: {path}')
    for record in removed:
        canonical = checked_path(record['retained_path'])
        path = checked_path(record['original_path'])
        if sha(canonical) != sha(path):
            raise RuntimeError(f'Duplicate changed before removal: {path}')
        path.unlink()
    for record in retained:
        assert sha(checked_path(record['original_path'])) == record['sha256']
    plan['status'] = 'completed'
    args.plan.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'COMPLETE: removed {len(removed)} verified duplicates; all {len(retained)} unique hashes retained')
    print('Restoration: copy retained_path to original_path for each duplicate_files entry.')


if __name__ == '__main__':
    main()
