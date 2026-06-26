# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `79808`, seconds `10.50`, LSTM `0.0887`, delta `-0.2207`
- tick `79776`, seconds `10.00`, LSTM `0.3094`, delta `-0.1845`
- tick `79648`, seconds `8.00`, LSTM `0.4655`, delta `-0.0264`
- tick `80032`, seconds `14.00`, LSTM `0.0517`, delta `-0.0175`
- tick `79712`, seconds `9.00`, LSTM `0.4824`, delta `+0.0157`
- tick `79968`, seconds `13.00`, LSTM `0.0670`, delta `-0.0143`
- tick `81184`, seconds `32.00`, LSTM `0.0065`, delta `-0.0137`
- tick `79872`, seconds `11.50`, LSTM `0.0900`, delta `+0.0133`
- tick `79584`, seconds `7.00`, LSTM `0.5015`, delta `+0.0130`
- tick `79840`, seconds `11.00`, LSTM `0.0767`, delta `-0.0120`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001973`, |coef| `0.001973`
- `lag_13__CT_place_HOUSE`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_07__CT_flashes_last_5s`: coefficient `0.001596`, |coef| `0.001596`
- `lag_08__CT_flashes_last_5s`: coefficient `0.001584`, |coef| `0.001584`
- `lag_05__CT4__flash_duration`: coefficient `-0.001494`, |coef| `0.001494`
- `lag_03__T_burning_players`: coefficient `-0.001337`, |coef| `0.001337`
- `lag_06__CT4__flash_duration`: coefficient `-0.001309`, |coef| `0.001309`
- `lag_04__CT_place_TOPOFMID`: coefficient `0.001238`, |coef| `0.001238`
- `lag_05__CT_place_TOPOFMID`: coefficient `0.001203`, |coef| `0.001203`
- `lag_06__CT_place_TOPOFMID`: coefficient `-0.001156`, |coef| `0.001156`
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.001058`, |coef| `0.001058`
- `lag_00__T_kills_last_3s`: coefficient `-0.001007`, |coef| `0.001007`
- `lag_00__T_damage_last_5s`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_02__T_burning_players`: coefficient `-0.000934`, |coef| `0.000934`
- `lag_01__T_shots_fired_sum`: coefficient `-0.000931`, |coef| `0.000931`

## Top 10 utility ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.001596` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.001584` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.001494` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.001309` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `-0.000732` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000693` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000683` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.000666` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000653` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.000648` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001973` (lowers CT win probability)
- `lag_13__CT_place_HOUSE`: coefficient `-0.001688` (lowers CT win probability)
- `lag_03__T_burning_players`: coefficient `-0.001337` (lowers CT win probability)
- `lag_04__CT_place_TOPOFMID`: coefficient `0.001238` (raises CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `0.001203` (raises CT win probability)
- `lag_06__CT_place_TOPOFMID`: coefficient `-0.001156` (lowers CT win probability)
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.001058` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001007` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000972` (lowers CT win probability)
- `lag_02__T_burning_players`: coefficient `-0.000934` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `79808`, seconds `10.50`, LSTM delta `-0.2207`

Top all feature movements:
- `lag_13__CT_place_HOUSE`: contribution `-0.017888`
- `lag_08__CT_flashes_last_5s`: contribution `-0.017412`
- `lag_00__T_shots_fired_sum`: contribution `-0.014796`
- `lag_05__CT_place_TOPOFMID`: contribution `-0.008734`
- `lag_07__CT_place_TOPOFMID`: contribution `-0.007681`

Top utility-only movements:
- `lag_08__CT_flashes_last_5s`: contribution `-0.017412`
- `lag_06__CT4__flash_duration`: contribution `-0.007010`
- `lag_00__CT4__flash_duration`: contribution `-0.003243`
- `lag_00__CT3__flash_duration`: contribution `-0.002906`

### tick `79776`, seconds `10.00`, LSTM delta `-0.1845`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.017548`
- `lag_00__T_shots_fired_sum`: contribution `-0.011837`
- `lag_04__CT_place_TOPOFMID`: contribution `-0.008988`
- `lag_06__CT_place_TOPOFMID`: contribution `-0.008387`
- `lag_05__CT4__flash_duration`: contribution `-0.007999`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.017548`
- `lag_05__CT4__flash_duration`: contribution `-0.007999`
- `lag_00__CT4__flash_duration`: contribution `+0.002802`
- `lag_02__CT3__flash_duration`: contribution `-0.002715`

### tick `79648`, seconds `8.00`, LSTM delta `-0.0264`

Top all feature movements:
- `lag_08__CT_place_HOUSE`: contribution `-0.004472`
- `lag_13__CT_flashes_last_5s`: contribution `-0.003500`
- `lag_00__CT_place_MIDDLE`: contribution `+0.003410`
- `lag_03__CT_flashes_last_5s`: contribution `-0.003333`
- `lag_07__CT_place_HOUSE`: contribution `-0.002730`

Top utility-only movements:
- `lag_13__CT_flashes_last_5s`: contribution `-0.003500`
- `lag_03__CT_flashes_last_5s`: contribution `-0.003333`
- `lag_03__T_B_site_active_smokes`: contribution `-0.000945`
- `lag_01__CT_flash_duration_sum`: contribution `+0.000710`
- `lag_00__T_A_site_active_infernos`: contribution `-0.000574`

### tick `80032`, seconds `14.00`, LSTM delta `-0.0175`

Top all feature movements:
- `lag_07__CT4__flash_duration`: contribution `-0.003399`
- `lag_14__CT_place_HOUSE`: contribution `+0.003234`
- `lag_13__CT4__flash_duration`: contribution `-0.003122`
- `lag_15__CT_flashes_last_5s`: contribution `-0.002977`
- `lag_08__CT4__flash_duration`: contribution `+0.001990`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.003399`
- `lag_13__CT4__flash_duration`: contribution `-0.003122`
- `lag_15__CT_flashes_last_5s`: contribution `-0.002977`
- `lag_08__CT4__flash_duration`: contribution `+0.001990`

### tick `79712`, seconds `9.00`, LSTM delta `+0.0157`

Top all feature movements:
- `lag_04__CT_place_TOPOFMID`: contribution `+0.008988`
- `lag_00__CT_flashed_players`: contribution `+0.003254`
- `lag_15__CT_flashes_last_5s`: contribution `+0.002977`
- `lag_00__CT3__flash_duration`: contribution `+0.002906`
- `lag_02__T_flashed_players`: contribution `+0.002584`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.002977`
- `lag_00__CT3__flash_duration`: contribution `+0.002906`
- `lag_03__CT4__flash_duration`: contribution `-0.001886`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001511`
- `lag_05__CT_flashes_last_5s`: contribution `+0.001201`
