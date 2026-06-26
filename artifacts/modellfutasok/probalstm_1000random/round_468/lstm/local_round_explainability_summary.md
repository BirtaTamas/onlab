# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `21471`, seconds `32.50`, LSTM `0.1690`, delta `-0.1961`
- tick `22975`, seconds `56.00`, LSTM `0.0646`, delta `-0.1166`
- tick `21311`, seconds `30.00`, LSTM `0.3076`, delta `+0.0885`
- tick `20991`, seconds `25.00`, LSTM `0.4070`, delta `-0.0856`
- tick `21055`, seconds `26.00`, LSTM `0.3282`, delta `-0.0794`
- tick `22623`, seconds `50.50`, LSTM `0.1062`, delta `+0.0773`
- tick `21567`, seconds `34.00`, LSTM `0.0423`, delta `-0.0726`
- tick `20863`, seconds `23.00`, LSTM `0.4782`, delta `+0.0470`
- tick `20127`, seconds `11.50`, LSTM `0.4010`, delta `-0.0469`
- tick `21151`, seconds `27.50`, LSTM `0.2441`, delta `-0.0445`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.002771`, |coef| `0.002771`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002307`, |coef| `0.002307`
- `lag_00__T_damage_last_5s`: coefficient `-0.001650`, |coef| `0.001650`
- `lag_13__CT_place_TSIDELOWER`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001459`, |coef| `0.001459`
- `lag_10__T5__flash_duration`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_00__T_kills_last_3s`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_02__T5__is_walking`: coefficient `-0.001198`, |coef| `0.001198`
- `lag_00__CT3__flash_duration`: coefficient `0.001179`, |coef| `0.001179`
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.001170`, |coef| `0.001170`
- `lag_15__T_place_RAMP`: coefficient `-0.001168`, |coef| `0.001168`
- `lag_05__CT3__flash_duration`: coefficient `-0.001101`, |coef| `0.001101`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_10__CT4__flash_duration`: coefficient `-0.001080`, |coef| `0.001080`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001078`, |coef| `0.001078`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002307` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001459` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `-0.001327` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001179` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.001101` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001080` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001078` (lowers CT win probability)
- `lag_02__T4__molly`: coefficient `-0.000715` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.000710` (lowers CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.000702` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.002771` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001650` (lowers CT win probability)
- `lag_13__CT_place_TSIDELOWER`: coefficient `-0.001472` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001233` (lowers CT win probability)
- `lag_02__T5__is_walking`: coefficient `-0.001198` (lowers CT win probability)
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.001170` (lowers CT win probability)
- `lag_15__T_place_RAMP`: coefficient `-0.001168` (lowers CT win probability)
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001099` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001067` (lowers CT win probability)
- `lag_04__T_place_SIDEENTRANCE`: coefficient `-0.001066` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21471`, seconds `32.50`, LSTM delta `-0.1961`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.027665`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.011063`
- `lag_10__T5__flash_duration`: contribution `-0.008143`
- `lag_05__CT3__flash_duration`: contribution `-0.006063`
- `lag_10__CT4__flash_duration`: contribution `-0.005426`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.027665`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.011063`
- `lag_10__T5__flash_duration`: contribution `-0.008143`
- `lag_05__CT3__flash_duration`: contribution `-0.006063`
- `lag_10__CT4__flash_duration`: contribution `-0.005426`

### tick `22975`, seconds `56.00`, LSTM delta `-0.1166`

Top all feature movements:
- `lag_12__CT_place_TSIDELOWER`: contribution `-0.037648`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.007741`
- `lag_02__CT_place_TSIDEUPPER`: contribution `-0.006648`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.004918`
- `lag_00__T_damage_last_5s`: contribution `-0.003956`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21311`, seconds `30.00`, LSTM delta `+0.0885`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `+0.006496`
- `lag_00__CT_place_SIDEHALL`: contribution `+0.005003`
- `lag_00__T_damage_last_5s`: contribution `+0.003956`
- `lag_10__T_place_TSIDELOWER`: contribution `+0.003952`
- `lag_05__T5__flash_duration`: contribution `+0.003773`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.006496`
- `lag_05__T5__flash_duration`: contribution `+0.003773`
- `lag_05__CT4__flash_duration`: contribution `+0.003251`
- `lag_00__T1__flash_duration`: contribution `+0.001705`

### tick `20991`, seconds `25.00`, LSTM delta `-0.0856`

Top all feature movements:
- `lag_02__T_place_SIDEENTRANCE`: contribution `-0.005362`
- `lag_00__T_damage_last_5s`: contribution `-0.003956`
- `lag_00__T_kills_last_3s`: contribution `-0.003905`
- `lag_12__T4__duck_amount`: contribution `-0.003398`
- `lag_04__CT5__duck_amount`: contribution `-0.002713`

Top utility-only movements:
- `lag_00__T3__molly`: contribution `-0.001514`

### tick `21055`, seconds `26.00`, LSTM delta `-0.0794`

Top all feature movements:
- `lag_04__T_place_SIDEENTRANCE`: contribution `-0.005204`
- `lag_02__T_place_TSIDELOWER`: contribution `-0.003215`
- `lag_10__CT_place_RAMP`: contribution `-0.002616`
- `lag_06__T2__duck_amount`: contribution `-0.002472`
- `lag_00__T2__duck_amount`: contribution `-0.002208`

Top utility-only movements:
- No utility movement among the top local contributors.
