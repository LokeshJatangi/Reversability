#!/usr/bin/env python3
"""Preserve deduplicated lightweight evidence and inventory untouched raw files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


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
    parser.add_argument('--raw-targets', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    destination = args.output / 'submission-manifest.json'
    if destination.exists():
        raise RuntimeError('Submission manifest exists; preserve it before rebuilding')
    records = []
    seen_content = {}
    aliases = []

    def preserve(source, relative):
        target = args.output / relative
        digest = sha(source)
        if digest in seen_content:
            aliases.append({'source': str(source), 'canonical_path': seen_content[digest], 'sha256': digest})
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != digest:
                raise RuntimeError(f'Conflicting submission evidence: {target}')
        else:
            shutil.copyfile(source, target)
        assert sha(target) == digest
        seen_content[digest] = str(relative)
        records.append({'path': str(relative), 'source': str(source),
                        'bytes': source.stat().st_size, 'sha256': digest})

    for part in args.parts:
        root = part / 'artifacts'
        for source in sorted(root.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(root)
            if relative.parts[0] == 'data' and source.name not in ('manifest.json', 'tokenizer_config.json'):
                continue
            if source.suffix not in ('.json', '.jsonl', '.log') and 'source-snapshots' not in relative.parts:
                continue
            preserve(source, Path('recorded') / relative)
    for source in sorted(Path('runs/correctness').glob('*.json')):
        preserve(source, Path('local-correctness') / source.name)

    inventory = []
    for root in args.raw_targets:
        if not root.exists() or root.is_symlink():
            raise RuntimeError(f'Invalid raw target: {root}')
        paths = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
        for source in paths:
            if source.is_symlink():
                raise RuntimeError(f'Refusing unresolved symlink: {source}')
            inventory.append({'original_path': str(source), 'bytes': source.stat().st_size,
                              'sha256': sha(source)})
    manifest = {
        'format_version': 1,
        'policy': 'Recorded evidence copied byte-for-byte and content-deduplicated; original raw weights, data and exports remain in place, ignored by Git, not moved or deleted.',
        'artifact_parts': [str(p) for p in args.parts],
        'submission_files': records,
        'duplicate_sources': aliases,
        'raw_storage_policy': 'Original relative paths preserved; ignored files are not present in Git history.',
        'raw_targets': [str(p) for p in args.raw_targets],
        'raw_files': inventory,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Preserved {len(records)} lightweight files; inventoried {len(inventory)} raw files / {sum(r["bytes"] for r in inventory):,} bytes')
    print(f'Deduplicated {len(aliases)} sources; original raw files left untouched')


if __name__ == '__main__':
    main()
