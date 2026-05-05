# Utility Delta By No-Utility Uncertainty

- split: `test`
- threshold: `0.5`
- rows: `801393`
- no-utility bizonytalan sorok, margin <= 0.05: `156833`

| no-utility margin bin | rows | mean abs delta | p95 abs delta | good direction | mean logloss improvement | class flip rate | with acc | no acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `very_uncertain_0_02` | 71551 | 0.007137 | 0.019570 | 0.5162 | 0.000200 | 0.1825 | 0.5190 | 0.5174 |
| `uncertain_0_05` | 85282 | 0.007795 | 0.020928 | 0.5036 | 0.000138 | 0.0084 | 0.5384 | 0.5383 |
| `mild_uncertain_0_10` | 86893 | 0.008316 | 0.022851 | 0.4981 | -0.000423 | 0.0001 | 0.5987 | 0.5987 |
| `medium_0_20` | 114086 | 0.009107 | 0.025447 | 0.5067 | 0.001033 | 0.0000 | 0.6882 | 0.6882 |
| `confident_gt_0_20` | 443581 | 0.004530 | 0.015884 | 0.3454 | -0.000845 | 0.0000 | 0.8966 | 0.8966 |

## Ertelmezes

A `no_utility_margin` azt meri, hogy a no-utility modell probabilityje milyen messze van a thresholdtol.
Minel kisebb ez az ertek, annal bizonytalanabb a no-utility modell class dontese.

Ha a utility feature-ok foleg a bizonytalan sorokban mozditanak nagyobbat, akkor ez jo erv arra,
hogy a utility nem globalis accuracy-javulaskent, hanem hatarhelyzeti plusz informaciokent hasznos.
