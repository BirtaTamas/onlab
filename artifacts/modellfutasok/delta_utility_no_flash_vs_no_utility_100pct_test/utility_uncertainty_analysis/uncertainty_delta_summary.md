# Utility Delta By No-Utility Uncertainty

- split: `test`
- threshold: `0.5`
- rows: `801393`
- no-utility bizonytalan sorok, margin <= 0.05: `156833`

| no-utility margin bin | rows | mean abs delta | p95 abs delta | good direction | mean logloss improvement | class flip rate | with acc | no acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `very_uncertain_0_02` | 71551 | 0.007533 | 0.020170 | 0.5235 | 0.000752 | 0.1932 | 0.5281 | 0.5174 |
| `uncertain_0_05` | 85282 | 0.008319 | 0.023556 | 0.5109 | 0.000526 | 0.0119 | 0.5386 | 0.5383 |
| `mild_uncertain_0_10` | 86893 | 0.009380 | 0.026411 | 0.4784 | -0.000830 | 0.0002 | 0.5986 | 0.5987 |
| `medium_0_20` | 114086 | 0.010056 | 0.027892 | 0.4735 | 0.000630 | 0.0000 | 0.6882 | 0.6882 |
| `confident_gt_0_20` | 443581 | 0.004759 | 0.016849 | 0.3353 | -0.001230 | 0.0000 | 0.8966 | 0.8966 |

## Ertelmezes

A `no_utility_margin` azt meri, hogy a no-utility modell probabilityje milyen messze van a thresholdtol.
Minel kisebb ez az ertek, annal bizonytalanabb a no-utility modell class dontese.

Ha a utility feature-ok foleg a bizonytalan sorokban mozditanak nagyobbat, akkor ez jo erv arra,
hogy a utility nem globalis accuracy-javulaskent, hanem hatarhelyzeti plusz informaciokent hasznos.
