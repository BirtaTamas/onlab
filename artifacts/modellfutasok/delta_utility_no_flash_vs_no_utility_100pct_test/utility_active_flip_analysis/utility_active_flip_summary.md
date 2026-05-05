# Utility Active Class Flip Analysis

Ez az elemzes csak azokat a snapshotokat nezi, ahol a with-utility es no-utility modell 0.5 threshold mellett mas class-t prediktalt.
A csoportok azt mutatjak, hogy milyen utility jel volt jelen az adott atbillenesnel.

| csoport | sorok | utility jo | no-utility jo | kulonbseg | utility win rate | mean abs delta | mean delta | CT win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_flips` | 14855 | 7822 | 7033 | 789 | 52.66% | 0.014133 | 0.000626 | 50.04% |
| `any_utility_nonzero` | 14855 | 7822 | 7033 | 789 | 52.66% | 0.014133 | 0.000626 | 50.04% |
| `active_or_recent_utility` | 10548 | 5579 | 4969 | 610 | 52.89% | 0.014416 | 0.000866 | 49.91% |
| `strong_utility_action` | 10244 | 5426 | 4818 | 608 | 52.97% | 0.014538 | 0.000809 | 50.07% |
| `utility_damage` | 1537 | 798 | 739 | 59 | 51.92% | 0.015189 | 0.001101 | 49.32% |
| `active_smoke_or_inferno` | 9794 | 5226 | 4568 | 658 | 53.36% | 0.014488 | 0.000756 | 50.65% |
| `recent_utility_last_5s` | 2075 | 1078 | 997 | 81 | 51.95% | 0.013895 | 0.001800 | 49.16% |
| `flash_effect` | 3161 | 1566 | 1595 | -29 | 49.54% | 0.015319 | -0.000028 | 47.90% |

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
