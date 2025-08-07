import re
import statistics
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# 1.  LOAD RAW LOG  ────────────────────────────────────────────────
# ------------------------------------------------------------------
#log_path = Path("/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/GPU_log/logCifar10Full18.txt")    
log_path = Path("/mnt/scratch/ralexander/ACCL/integrations/pytorch_ddp/test/GPU_log/logCifar10Full50.txt")     # change if your file name differs
log_text = log_path.read_text(encoding="utf‑8")

# ------------------------------------------------------------------
# 2.  SPLIT BY RUN (“CASE”) AND GATHER METRICS  ────────────────────
# ------------------------------------------------------------------
pattern_header = re.compile(r"^\s*(\d+\s+Node.*?):\s*$", re.MULTILINE)
pattern_epoch  = re.compile(
    r"Epoch\s*-\s*(\d+)/\d+\s*:\s*time\s*-\s*([\d.]+)s\s*\|\|"
    r"\s*loss_train\s*-\s*([\d.]+)\s*\|\|\s*accuracy_train\s*-\s*([\d.]+)",
    re.MULTILINE,
)
pattern_test   = re.compile(r"Accuracy on test dataset\s*-\s*([\d.]+)")

runs = []
for hdr_match, next_hdr in zip(
        pattern_header.finditer(log_text),
        list(pattern_header.finditer(log_text))[1:] + [None]
):
    start = hdr_match.end()
    end   = next_hdr.start() if next_hdr else len(log_text)
    block = log_text[start:end]

    header = hdr_match.group(1).rstrip(":")
    epochs = pattern_epoch.findall(block)
    tests  = pattern_test.findall(block)

    if not epochs:
        continue

    times = [float(e[1]) for e in epochs]
    total_time = sum(times)
    avg_time   = total_time / len(times)
    std_time   = statistics.stdev(times) if len(times) > 1 else 0.0

    runs.append({
        "Case": header,
        "Epochs": len(epochs),
        "Total Time (s)": total_time,
        "Avg Epoch Time (s)": avg_time,
        "Std Dev Epoch Time (s)": std_time,
        "Final Train Acc": float(epochs[-1][3]),
        "Final Test Acc": float(tests[-1]) if tests else None,
    })

# ------------------------------------------------------------------
# 3.  PRESENT RESULTS  ─────────────────────────────────────────────
# ------------------------------------------------------------------
df = pd.DataFrame(runs).sort_values("Case").reset_index(drop=True)
print(df.to_string(index=False, float_format="%.4f"))