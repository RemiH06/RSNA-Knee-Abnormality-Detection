"""Arma kaggle/bench/kad-knee-bench.py: el código de kaggle/infer_v2/kad-knee-infer.py sin su main
más bench_part.py (medición de tiempo y de AUC contra K sobre los 58 gold)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "infer_v2" / "kad-knee-infer.py").read_text(encoding="utf-8").rstrip()
tail = 'if __name__ == "__main__":\n    main()'
assert src.endswith(tail), "cambió el final de kad-knee-infer.py"
src = src[: -len(tail)]
part = (HERE / "bench_part.py").read_text(encoding="utf-8")
(HERE / "kad-knee-bench.py").write_text(src + part, encoding="utf-8", newline="\n")
print("escrito kad-knee-bench.py")
