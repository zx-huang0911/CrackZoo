from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackzoo.data_audit import audit_dataset


def main():
    p = argparse.ArgumentParser(description="Check pairs and exact cross-split duplicates; keep report local.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    a = p.parse_args()
    result = audit_dataset(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"samples={len(result['samples'])}; errors={len(result['errors'])}; report={a.output}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
