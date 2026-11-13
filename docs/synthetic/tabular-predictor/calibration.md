# SYNTHETIC DATA. NOT A RESULT.

> **SYNTHETIC DATA. NOT A RESULT.** Made from the evaluator's synthetic generator.

## Calibration report

- Test tasks: 190; probes describing each configuration: 50
- Binning: 10 equal-width bins of the predicted probability
- Intervals: 95% percentile interval, 300 task-cluster bootstrap replicates
- Threshold window: rows with |predicted - tau| <= 0.05
- Calibrator: none
- Data fingerprint: `fb8f9faaa41067992eca579ab24006b96ee49136f324abf58e12a79827b3499d`

> SYNTHETIC DATA. NOT A RESULT.

### first step (no earlier failure)

1140 rows from 190 tasks; observed success rate 0.665; Brier score of the constant base rate 0.2228.

| metric | uncalibrated | 95% interval | calibrated | 95% interval |
|---|---|---|---|---|
| ECE | 0.0684 | [0.0469, 0.0979] | 0.0684 | [0.0469, 0.0979] |
| Brier | 0.1618 | [0.1487, 0.1765] | 0.1618 | [0.1487, 0.1765] |
| log loss | 0.5001 | [0.4703, 0.5334] | 0.5001 | [0.4703, 0.5334] |
| AUC | 0.8147 | | 0.8147 | |
| accuracy at 0.5 | 0.7868 | | 0.7868 | |

Calibrated minus uncalibrated (negative is better):

| metric | difference | 95% interval |
|---|---|---|
| brier | +0.0000 | [0.0000, 0.0000] |
| log_loss | +0.0000 | [0.0000, 0.0000] |
| ece | +0.0000 | [0.0000, 0.0000] |

Near the policy's thresholds (observed minus predicted success):

| tau | rows in window | uncalibrated | 95% interval | calibrated | 95% interval |
|---|---|---|---|---|---|
| 0.5 | 106 (calibrated), 106 (raw) | -0.062 | [-0.152, 0.031] | -0.062 | [-0.152, 0.031] |
| 0.7 | 95 (calibrated), 95 (raw) | 0.059 | [-0.027, 0.146] | 0.059 | [-0.027, 0.146] |
| 0.8 | 567 (calibrated), 567 (raw) | 0.061 | [0.025, 0.099] | 0.061 | [0.025, 0.099] |
| 0.9 | 0 (calibrated), 0 (raw) | n/a | n/a | n/a | n/a |

Reliability table (calibrated):

| predicted bin | rows | mean predicted | observed |
|---|---|---|---|
| [0.3, 0.4) | 208 | 0.328 | 0.250 |
| [0.4, 0.5) | 119 | 0.459 | 0.353 |
| [0.5, 0.6) | 59 | 0.532 | 0.559 |
| [0.6, 0.7) | 159 | 0.653 | 0.723 |
| [0.7, 0.8) | 246 | 0.766 | 0.801 |
| [0.8, 0.9) | 349 | 0.834 | 0.914 |

> SYNTHETIC DATA. NOT A RESULT.

### second step, after one failed attempt

1213 rows from 134 tasks; observed success rate 0.551; Brier score of the constant base rate 0.2474.

| metric | uncalibrated | 95% interval | calibrated | 95% interval |
|---|---|---|---|---|
| ECE | 0.0617 | [0.0473, 0.1121] | 0.0617 | [0.0473, 0.1121] |
| Brier | 0.1968 | [0.1784, 0.2158] | 0.1968 | [0.1784, 0.2158] |
| log loss | 0.5811 | [0.5384, 0.6240] | 0.5811 | [0.5384, 0.6240] |
| AUC | 0.7708 | | 0.7708 | |
| accuracy at 0.5 | 0.7279 | | 0.7279 | |

Calibrated minus uncalibrated (negative is better):

| metric | difference | 95% interval |
|---|---|---|
| brier | +0.0000 | [0.0000, 0.0000] |
| log_loss | +0.0000 | [0.0000, 0.0000] |
| ece | +0.0000 | [0.0000, 0.0000] |

Near the policy's thresholds (observed minus predicted success):

| tau | rows in window | uncalibrated | 95% interval | calibrated | 95% interval |
|---|---|---|---|---|---|
| 0.5 | 174 (calibrated), 174 (raw) | -0.085 | [-0.196, 0.061] | -0.085 | [-0.196, 0.061] |
| 0.7 | 263 (calibrated), 263 (raw) | -0.018 | [-0.108, 0.072] | -0.018 | [-0.108, 0.072] |
| 0.8 | 327 (calibrated), 327 (raw) | -0.012 | [-0.102, 0.075] | -0.012 | [-0.102, 0.075] |
| 0.9 | 0 (calibrated), 0 (raw) | n/a | n/a | n/a | n/a |

Reliability table (calibrated):

| predicted bin | rows | mean predicted | observed |
|---|---|---|---|
| [0.3, 0.4) | 214 | 0.325 | 0.154 |
| [0.4, 0.5) | 247 | 0.459 | 0.364 |
| [0.5, 0.6) | 57 | 0.555 | 0.544 |
| [0.6, 0.7) | 208 | 0.651 | 0.678 |
| [0.7, 0.8) | 235 | 0.737 | 0.706 |
| [0.8, 0.9) | 252 | 0.828 | 0.821 |

> SYNTHETIC DATA. NOT A RESULT.

### Per configuration, first step

| configuration | held out | rows | observed | mean predicted | Brier | AUC within configuration |
|---|---|---|---|---|---|---|
| cloud-big (high) | no | 190 | 0.837 | 0.769 | 0.1157 | 0.765 |
| cloud-big (medium) | no | 190 | 0.789 | 0.761 | 0.1384 | 0.736 |
| cloud-mid (medium) | no | 190 | 0.726 | 0.691 | 0.1531 | 0.798 |
| cloud-small (high) | no | 190 | 0.642 | 0.597 | 0.1743 | 0.788 |
| cloud-small (low) | no | 190 | 0.526 | 0.558 | 0.1906 | 0.782 |
| local-small (medium) | no | 190 | 0.468 | 0.505 | 0.1990 | 0.770 |

**SYNTHETIC DATA. NOT A RESULT.**
