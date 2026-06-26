# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `15`

## Largest probability jumps

- tick `122150`, seconds `0.50`, LSTM `0.0680`, delta `-0.0792`
- tick `124262`, seconds `33.50`, LSTM `0.0473`, delta `-0.0712`
- tick `122470`, seconds `5.50`, LSTM `0.1265`, delta `+0.0449`
- tick `122598`, seconds `7.50`, LSTM `0.1161`, delta `-0.0199`
- tick `122502`, seconds `6.00`, LSTM `0.1451`, delta `+0.0186`
- tick `122662`, seconds `8.50`, LSTM `0.0916`, delta `-0.0171`
- tick `122758`, seconds `10.00`, LSTM `0.1245`, delta `+0.0153`
- tick `124006`, seconds `29.50`, LSTM `0.1364`, delta `+0.0144`
- tick `122406`, seconds `4.50`, LSTM `0.0692`, delta `+0.0144`
- tick `124390`, seconds `35.50`, LSTM `0.0147`, delta `-0.0140`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001544`, |coef| `0.001544`
- `lag_00__T_flashes_last_5s`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_08__T5__flash_duration`: coefficient `-0.000610`, |coef| `0.000610`
- `lag_00__CT_place_MAIN`: coefficient `0.000591`, |coef| `0.000591`
- `lag_01__T4__flash_duration`: coefficient `-0.000550`, |coef| `0.000550`
- `lag_08__T2__flash_duration`: coefficient `-0.000547`, |coef| `0.000547`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000530`, |coef| `0.000530`
- `lag_08__T_flashed_players`: coefficient `-0.000485`, |coef| `0.000485`
- `lag_08__T_flash_duration_sum`: coefficient `-0.000483`, |coef| `0.000483`
- `lag_11__CT_place_CTSIDEUPPER`: coefficient `0.000474`, |coef| `0.000474`
- `lag_07__CT_place_CTSIDEUPPER`: coefficient `-0.000416`, |coef| `0.000416`
- `lag_00__T_damage_last_5s`: coefficient `-0.000374`, |coef| `0.000374`
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `-0.000371`, |coef| `0.000371`
- `lag_00__T5__flash_duration`: coefficient `0.000367`, |coef| `0.000367`
- `lag_01__T_place_BRIDGE`: coefficient `-0.000355`, |coef| `0.000355`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000761` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000610` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.000550` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.000547` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.000483` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000367` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000283` (lowers CT win probability)
- `lag_11__T5__molly`: coefficient `0.000259` (raises CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000252` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.000246` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001544` (lowers CT win probability)
- `lag_00__CT_place_MAIN`: coefficient `0.000591` (raises CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000530` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `-0.000485` (lowers CT win probability)
- `lag_11__CT_place_CTSIDEUPPER`: coefficient `0.000474` (raises CT win probability)
- `lag_07__CT_place_CTSIDEUPPER`: coefficient `-0.000416` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000374` (lowers CT win probability)
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `-0.000371` (lowers CT win probability)
- `lag_01__T_place_BRIDGE`: coefficient `-0.000355` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `-0.000351` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `122150`, seconds `0.50`, LSTM delta `-0.0792`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.039779`
- `lag_00__T_flashes_last_5s`: contribution `-0.006893`
- `lag_01__T_place_TSPAWN`: contribution `-0.001446`
- `lag_00__T_velocity_mean`: contribution `-0.000998`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000725`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.006893`
- `lag_01__CT1__flash`: contribution `-0.000546`
- `lag_01__molly_inv_diff`: contribution `-0.000521`
- `lag_00__CT4__smoke`: contribution `-0.000425`
- `lag_01__T_smoke_inv`: contribution `-0.000423`

### tick `124262`, seconds `33.50`, LSTM delta `-0.0712`

Top all feature movements:
- `lag_08__T5__flash_duration`: contribution `-0.004499`
- `lag_00__CT_place_MAIN`: contribution `-0.003982`
- `lag_01__T4__flash_duration`: contribution `-0.003720`
- `lag_08__T2__flash_duration`: contribution `-0.002873`
- `lag_08__T_flashed_players`: contribution `-0.002810`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.004499`
- `lag_01__T4__flash_duration`: contribution `-0.003720`
- `lag_08__T2__flash_duration`: contribution `-0.002873`
- `lag_08__T_flash_duration_sum`: contribution `-0.002729`

### tick `122470`, seconds `5.50`, LSTM delta `+0.0449`

Top all feature movements:
- `lag_11__CT_place_CTSIDEUPPER`: contribution `+0.012200`
- `lag_00__T_flashes_last_5s`: contribution `+0.006893`
- `lag_06__CT_place_CTSIDEUPPER`: contribution `+0.002976`
- `lag_01__CT_place_PALACEINTERIOR`: contribution `+0.002523`
- `lag_01__T_place_STREET`: contribution `+0.002345`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.006893`
- `lag_10__T_flashes_last_5s`: contribution `+0.002284`
- `lag_11__CT1__flash`: contribution `+0.000696`
- `lag_11__T5__molly`: contribution `+0.000409`
- `lag_11__molly_inv_diff`: contribution `+0.000264`

### tick `122598`, seconds `7.50`, LSTM delta `-0.0199`

Top all feature movements:
- `lag_15__CT_place_CTSIDEUPPER`: contribution `-0.004318`
- `lag_09__CT_place_CTSIDEUPPER`: contribution `-0.001975`
- `lag_05__CT_place_PALACEINTERIOR`: contribution `-0.001631`
- `lag_00__CT_place_WALKWAY`: contribution `-0.001376`
- `lag_10__CT_place_CTSIDEUPPER`: contribution `-0.001332`

Top utility-only movements:
- `lag_04__T_flashes_last_5s`: contribution `-0.000949`

### tick `122502`, seconds `6.00`, LSTM delta `+0.0186`

Top all feature movements:
- `lag_07__CT_place_CTSIDEUPPER`: contribution `+0.004296`
- `lag_12__CT_place_CTSIDEUPPER`: contribution `+0.003356`
- `lag_06__CT_place_CTSIDEUPPER`: contribution `+0.002976`
- `lag_01__T_flashes_last_5s`: contribution `+0.001755`
- `lag_06__CT_place_PALACEINTERIOR`: contribution `+0.001638`

Top utility-only movements:
- `lag_01__T_flashes_last_5s`: contribution `+0.001755`
- `lag_12__CT1__flash`: contribution `+0.000281`
- `lag_01__T3__molly`: contribution `+0.000228`
- `lag_11__T_flashes_last_5s`: contribution `+0.000221`
