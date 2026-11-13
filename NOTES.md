# NOTES - Attempts and Fixes

## Run
- Date: Nov 2026, Single GPU, Python 3.10, PyTorch
- Run: 2 epochs, batch 32, seed 0, n=160 (80 Live, 80 Attack)
- Result: Loss 0.1390 -> 0.0002, ACER 0.00%, Acc 100%, AUC 100%

## Attempt 1 - Docker
- Tried: README says docker run --gpus all
- Outcome: FAILED - No GPU Docker available. Ran natively instead.

## Attempt 2 - Data Path
- Tried: -v ../data:/data
- Outcome: FAILED - Path not found. Fixed with absolute path.

## Attempt 3 - Dataset
- Tried: Download from repo
- Outcome: FAILED - Dataset not in repo. Needed UniAttackData challenge access.

## Attempt 4 - Metric Mismatch
- Found: README says EER 2.10, Paper says ACER 2.10
- Outcome: Used ACER in report, noted mismatch.

## Attempt 5 - Table Mismatch
- Found: Table 2 vs Table 3 give 0.0085 vs 0.0108 for same threshold
- Outcome: Used validation ACER 3.6% as reference, noted mismatch.

## Attempt 6 - Environment
- Found: No Python version, epochs, expected output documented
- Outcome: Created SETUP.md + requirements.txt

## Note on Seeds
- Reduced to 1 seed, 2 epochs, n=160 due to limited Colab time. Reported as single run, not mean ± std.
