# Recorded evidence

This directory is the lightweight, versioned evidence accompanying the main study report. Original full/smoke metric streams, summaries, console/orchestration logs, capacity reports, frozen configurations, correctness gates and source snapshots are copied byte-for-byte from the final split exports.

- `recorded/` mirrors their original artifact-relative layout.
- `local-correctness/` includes earlier CPU reports, including the failed first numerical policy.
- `submission-manifest.json` lists copied-file SHA-256 hashes, source locations and the original pre-cleanup raw-file inventory.
- `raw-deduplication.json` records completed raw cleanup and maps each removed duplicate path to its surviving identical `retained_path`.

There is one copy of each distinct lightweight evidence file. Dataset binaries and model/optimizer checkpoints are not duplicated here or committed to Git. Raw cleanup removed 80 checksum-identical redundant files, preserving every unique raw file (96) and reclaiming 2,725,406,564 bytes. The final split export remains complete; older folders may be partial, with removed paths recoverable by copying the recorded retained file. Git history does not preserve ignored files; Google Drive was not modified. The inherited `recorded/artifact-inventory.json` describes an older baseline export, not the current two-part export's complete file set.

Recorded configs and commands retain their original Colab absolute paths. Do not rewrite these evidence files to adapt a new run; create a new configuration instead. Both final raw export parts are needed for the full checkpoint audit, because two checkpoints are stored in part 002.

The main [study report](../README.md) explains results and reproduction. The [structured audit](../docs/analysis/core-audit-2026-10-06.json) covers all seven smoke/full streams and ten reversible checkpoints. The full trajectories can be regenerated solely from this directory with:

```bash
python3 docs/analysis/plot_matched_results.py --artifacts results/recorded --include-maximum --output /tmp/reversibility-loss.svg
```
