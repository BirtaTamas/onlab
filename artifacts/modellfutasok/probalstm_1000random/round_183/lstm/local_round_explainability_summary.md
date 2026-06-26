# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `20`

## Largest probability jumps

- tick `154744`, seconds `46.50`, LSTM `0.8865`, delta `+0.0980`
- tick `153080`, seconds `20.50`, LSTM `0.7586`, delta `-0.0485`
- tick `153208`, seconds `22.50`, LSTM `0.7952`, delta `+0.0334`
- tick `152632`, seconds `13.50`, LSTM `0.8056`, delta `+0.0289`
- tick `154616`, seconds `44.50`, LSTM `0.7774`, delta `-0.0287`
- tick `154296`, seconds `39.50`, LSTM `0.8018`, delta `+0.0260`
- tick `153240`, seconds `23.00`, LSTM `0.7711`, delta `-0.0241`
- tick `156408`, seconds `72.50`, LSTM `0.9351`, delta `+0.0240`
- tick `154168`, seconds `37.50`, LSTM `0.7703`, delta `-0.0238`
- tick `156632`, seconds `76.00`, LSTM `0.9454`, delta `+0.0191`

## Top 15 local ridge features

- `lag_14__CT_place_EXTENDEDA`: coefficient `0.000684`, |coef| `0.000684`
- `lag_00__CT_place_SIDE`: coefficient `0.000633`, |coef| `0.000633`
- `lag_04__T_flashed_players`: coefficient `0.000597`, |coef| `0.000597`
- `lag_00__T4__is_walking`: coefficient `-0.000593`, |coef| `0.000593`
- `lag_04__T5__flash_duration`: coefficient `0.000586`, |coef| `0.000586`
- `lag_11__CT1__is_scoped`: coefficient `-0.000548`, |coef| `0.000548`
- `lag_04__CT_place_SIDE`: coefficient `-0.000546`, |coef| `0.000546`
- `lag_00__CT_kills_last_3s`: coefficient `0.000527`, |coef| `0.000527`
- `lag_13__CT_place_EXTENDEDA`: coefficient `0.000516`, |coef| `0.000516`
- `lag_00__T4__alive`: coefficient `-0.000514`, |coef| `0.000514`
- `lag_01__CT_place_MIDDOORS`: coefficient `-0.000511`, |coef| `0.000511`
- `lag_00__T4__hp`: coefficient `-0.000504`, |coef| `0.000504`
- `lag_03__T1__duck_amount`: coefficient `0.000497`, |coef| `0.000497`
- `lag_00__CT_place_MIDDOORS`: coefficient `-0.000482`, |coef| `0.000482`
- `lag_00__T4__armor`: coefficient `-0.000479`, |coef| `0.000479`

## Top 10 utility ridge features

- `lag_04__T5__flash_duration`: coefficient `0.000586` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000409` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000331` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.000319` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000310` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.000291` (raises CT win probability)
- `lag_08__CT1__flash`: coefficient `-0.000283` (lowers CT win probability)
- `lag_15__CT_active_smokes`: coefficient `-0.000256` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000227` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000217` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_EXTENDEDA`: coefficient `0.000684` (raises CT win probability)
- `lag_00__CT_place_SIDE`: coefficient `0.000633` (raises CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.000597` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000593` (lowers CT win probability)
- `lag_11__CT1__is_scoped`: coefficient `-0.000548` (lowers CT win probability)
- `lag_04__CT_place_SIDE`: coefficient `-0.000546` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000527` (raises CT win probability)
- `lag_13__CT_place_EXTENDEDA`: coefficient `0.000516` (raises CT win probability)
- `lag_00__T4__alive`: coefficient `-0.000514` (lowers CT win probability)
- `lag_01__CT_place_MIDDOORS`: coefficient `-0.000511` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `154744`, seconds `46.50`, LSTM delta `+0.0980`

Top all feature movements:
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.003840`
- `lag_04__T_flashed_players`: contribution `+0.003457`
- `lag_04__T5__flash_duration`: contribution `+0.003187`
- `lag_11__CT1__is_scoped`: contribution `+0.002348`
- `lag_03__T1__duck_amount`: contribution `+0.001944`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `+0.003187`

### tick `153080`, seconds `20.50`, LSTM delta `-0.0485`

Top all feature movements:
- `lag_00__CT_place_SIDE`: contribution `-0.020628`
- `lag_14__CT_place_SIDE`: contribution `-0.011516`
- `lag_08__CT_place_HOLE`: contribution `-0.003673`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.001862`
- `lag_00__CT3__duck_amount`: contribution `-0.001400`

Top utility-only movements:
- `lag_14__CT_B_site_active_infernos`: contribution `-0.000430`

### tick `153208`, seconds `22.50`, LSTM delta `+0.0334`

Top all feature movements:
- `lag_04__CT_place_SIDE`: contribution `+0.017788`
- `lag_12__CT_place_HOLE`: contribution `-0.001881`
- `lag_00__T4__is_walking`: contribution `+0.001368`
- `lag_03__T1__duck_amount`: contribution `+0.001039`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `-0.000985`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `152632`, seconds `13.50`, LSTM delta `+0.0289`

Top all feature movements:
- `lag_00__CT_place_SIDE`: contribution `+0.020628`
- `lag_08__CT_place_HOLE`: contribution `+0.003673`
- `lag_10__T5__is_scoped`: contribution `+0.001493`
- `lag_13__T_place_TOPOFMID`: contribution `+0.000771`
- `lag_02__CT1__is_walking`: contribution `+0.000729`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154616`, seconds `44.50`, LSTM delta `-0.0287`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `-0.001846`
- `lag_10__CT1__is_scoped`: contribution `-0.001741`
- `lag_00__T_flashed_players`: contribution `-0.001528`
- `lag_10__T5__is_scoped`: contribution `-0.001493`
- `lag_00__CT_place_BDOORS`: contribution `-0.001424`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.001117`
- `lag_00__T5__flash_duration`: contribution `-0.000865`
