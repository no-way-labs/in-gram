"""Check that the tokenized dataset on this machine is the one the paper used.

    python verify_data.py            # compares $TEXT_CACHE (default ./data) with data_manifest.json
    python verify_data.py --write    # (maintainers) writes data_manifest.json from the local cache

The dataset is built on first use by src/agentic.py: it streams the first 16,384 rows of
nebius/SWE-rebench-openhands-trajectories from Hugging Face, renders each trajectory as text, tokenizes it with
the GPT-2 vocabulary and splits the trajectories by repository. data_manifest.json holds the counts and the
SHA-256 of every token file of the cache built by that code on 2026-11-08 on the machine that ran the analyses.
Each training host built its own cache in the same way, and the slot counts that the runs logged for the
evaluation windows agree with the ones computed from this cache. If the upstream dataset changes, the hashes will
differ, and results on the new data are not comparable token for token.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "agentic-SWE-rebench-openhands-trajectories-gpt2-sys0-ev0.05-n16384"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cache = Path(os.environ.get("TEXT_CACHE", HERE / "data")) / NAME
    if not (cache / "meta.json").exists():
        sys.exit(f"no finished cache at {cache}; run a training or analysis script once to build it")
    local = dict(meta=json.loads((cache / "meta.json").read_text()),
                 files={f.name: dict(bytes=f.stat().st_size, sha256=sha256(f)) for f in sorted(cache.glob("*.bin"))})
    manifest = HERE / "data_manifest.json"
    if "--write" in sys.argv:
        manifest.write_text(json.dumps(local, indent=1) + "\n")
        print(f"wrote {manifest} ({len(local['files'])} token files)")
        return
    ref = json.loads(manifest.read_text())
    bad = [n for n in ref["files"] if local["files"].get(n) != ref["files"][n]]
    if local["meta"]["splits"] != ref["meta"]["splits"] or bad:
        print("the local dataset differs from the paper's:")
        print("  splits here :", local["meta"]["splits"])
        print("  splits paper:", ref["meta"]["splits"])
        print("  token files that differ:", bad or "none")
        sys.exit(1)
    print(f"the local dataset matches the paper's: {len(ref['files'])} token files, "
          f"{ref['meta']['splits']['train']['trajectories']} training and "
          f"{ref['meta']['splits']['eval']['trajectories']} held-out trajectories")


if __name__ == "__main__":
    main()
