# Utility Active Class Flip Analysis

Ez az elemzes csak azokat a snapshotokat nezi, ahol a with-utility es no-utility modell 0.5 threshold mellett mas class-t prediktalt.
A csoportok azt mutatjak, hogy milyen utility jel volt jelen az adott atbillenesnel.

| csoport | sorok | utility jo | no-utility jo | kulonbseg | utility win rate | mean abs delta | mean delta | CT win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_flips` | 13781 | 6951 | 6830 | 121 | 50.44% | 0.013171 | 0.001721 | 48.81% |
| `any_utility_nonzero` | 13781 | 6951 | 6830 | 121 | 50.44% | 0.013171 | 0.001721 | 48.81% |
| `active_or_recent_utility` | 9618 | 4839 | 4779 | 60 | 50.31% | 0.013634 | 0.002329 | 48.96% |
| `strong_utility_action` | 9320 | 4706 | 4614 | 92 | 50.49% | 0.013688 | 0.002246 | 49.33% |
| `utility_damage` | 1519 | 780 | 739 | 41 | 51.35% | 0.015357 | 0.003453 | 50.76% |
| `active_smoke_or_inferno` | 8872 | 4532 | 4340 | 192 | 51.08% | 0.013554 | 0.002071 | 49.98% |
| `recent_utility_last_5s` | 2004 | 1006 | 998 | 8 | 50.20% | 0.014056 | 0.003263 | 49.65% |
| `flash_effect` | 2963 | 1361 | 1602 | -241 | 45.93% | 0.015222 | 0.003605 | 43.33% |

## Csoportok jelentese

- `all_flips`: minden atbillenes.
- `any_utility_nonzero`: barmilyen utility feature nem nulla, inventory is beleszamit.
- `active_or_recent_utility`: aktiv smoke/inferno, last_5s utility, utility damage vagy flash duration latszik.
- `strong_utility_action`: erosebb utility-helyzet, peldaul tobb aktiv smoke/inferno, tobb recent utility event, legalabb 10 damage, vagy flash effect.
- `utility_damage`: volt utility sebzes az utolso 5 masodpercben.
- `active_smoke_or_inferno`: van aktiv smoke vagy inferno.
- `recent_utility_last_5s`: volt smoke/flash/he/molly az utolso 5 masodpercben.
- `flash_effect`: flash duration ertek alapjan valaki vakitva volt.

## Kimenetek

- `utility_active_flip_rows.csv`: minden atbillenes utility aktivitasi oszlopokkal.
- `top_utility_good_flips.csv`: legnagyobb jo iranyu atbillenesek.
- `top_utility_bad_flips.csv`: legnagyobb rossz iranyu atbillenesek.
