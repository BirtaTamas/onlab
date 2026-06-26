# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `28417`, seconds `66.00`, LSTM `0.1297`, delta `-0.2527`
- tick `27905`, seconds `58.00`, LSTM `0.5886`, delta `-0.1956`
- tick `27009`, seconds `44.00`, LSTM `0.6250`, delta `-0.1779`
- tick `28385`, seconds `65.50`, LSTM `0.3825`, delta `-0.1042`
- tick `27265`, seconds `48.00`, LSTM `0.9173`, delta `+0.0916`
- tick `27041`, seconds `44.50`, LSTM `0.7104`, delta `+0.0855`
- tick `28481`, seconds `67.00`, LSTM `0.0666`, delta `-0.0689`
- tick `27105`, seconds `45.50`, LSTM `0.7654`, delta `+0.0475`
- tick `27873`, seconds `57.50`, LSTM `0.7842`, delta `-0.0429`
- tick `27585`, seconds `53.00`, LSTM `0.8590`, delta `-0.0394`

## Top 15 local ridge features

- `lag_07__T_bomb_zone_count`: coefficient `0.003115`, |coef| `0.003115`
- `lag_00__T_kills_last_3s`: coefficient `-0.002935`, |coef| `0.002935`
- `lag_00__kill_diff_last_3s`: coefficient `0.002617`, |coef| `0.002617`
- `lag_07__T2__has_bomb`: coefficient `0.002187`, |coef| `0.002187`
- `lag_00__CT3__utility_total`: coefficient `0.002163`, |coef| `0.002163`
- `lag_06__T_bomb_zone_count`: coefficient `0.002126`, |coef| `0.002126`
- `lag_01__CT_A_site_active_infernos`: coefficient `-0.001996`, |coef| `0.001996`
- `lag_07__bomb_planted`: coefficient `-0.001977`, |coef| `0.001977`
- `lag_00__CT3__molly`: coefficient `0.001869`, |coef| `0.001869`
- `lag_00__CT3__alive`: coefficient `0.001867`, |coef| `0.001867`
- `lag_00__CT3__has_defuser`: coefficient `0.001747`, |coef| `0.001747`
- `lag_03__CT4__molly`: coefficient `0.001695`, |coef| `0.001695`
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.001667`, |coef| `0.001667`
- `lag_00__CT3__smoke`: coefficient `0.001636`, |coef| `0.001636`
- `lag_01__T_bomb_zone_count`: coefficient `-0.001618`, |coef| `0.001618`

## Top 10 utility ridge features

- `lag_00__CT3__utility_total`: coefficient `0.002163` (raises CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `-0.001996` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001869` (raises CT win probability)
- `lag_03__CT4__molly`: coefficient `0.001695` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.001667` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001636` (raises CT win probability)
- `lag_14__T4__molly`: coefficient `-0.001582` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001422` (raises CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `0.001354` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `-0.001292` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_bomb_zone_count`: coefficient `0.003115` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002935` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002617` (raises CT win probability)
- `lag_07__T2__has_bomb`: coefficient `0.002187` (raises CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `0.002126` (raises CT win probability)
- `lag_07__bomb_planted`: coefficient `-0.001977` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001867` (raises CT win probability)
- `lag_00__CT3__has_defuser`: coefficient `0.001747` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.001618` (lowers CT win probability)
- `lag_07__T_bomb_carrier_alive`: coefficient `0.001587` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28417`, seconds `66.00`, LSTM delta `-0.2527`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `-0.018135`
- `lag_01__T_place_BALCONY`: contribution `-0.011548`
- `lag_00__T_kills_last_3s`: contribution `-0.009298`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.007045`
- `lag_07__T2__has_bomb`: contribution `-0.006827`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `-0.007045`
- `lag_00__CT3__utility_total`: contribution `-0.006194`
- `lag_00__CT3__molly`: contribution `-0.004615`
- `lag_03__CT4__molly`: contribution `-0.004175`
- `lag_00__CT3__smoke`: contribution `-0.003619`

### tick `27905`, seconds `58.00`, LSTM delta `-0.1956`

Top all feature movements:
- `lag_10__T_place_BALCONY`: contribution `-0.010298`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009910`
- `lag_01__T_bomb_zone_count`: contribution `-0.009418`
- `lag_03__CT_place_LIBRARY`: contribution `-0.009392`
- `lag_00__T_kills_last_3s`: contribution `-0.009298`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009910`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.006600`
- `lag_00__CT2__molly`: contribution `-0.002584`

### tick `27009`, seconds `44.00`, LSTM delta `-0.1779`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.021471`
- `lag_00__T_kills_last_3s`: contribution `-0.009298`
- `lag_00__kill_diff_last_3s`: contribution `-0.006300`
- `lag_00__CT5__flash_duration`: contribution `-0.004788`
- `lag_12__CT5__duck_amount`: contribution `-0.003613`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.004788`
- `lag_10__CT3__molly`: contribution `-0.001895`
- `lag_00__CT1__molly`: contribution `-0.001508`

### tick `28385`, seconds `65.50`, LSTM delta `-0.1042`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.021471`
- `lag_06__T_bomb_zone_count`: contribution `-0.012377`
- `lag_15__CT_place_APARTMENTS`: contribution `-0.004634`
- `lag_06__T2__has_bomb`: contribution `-0.004251`
- `lag_06__bomb_planted`: contribution `-0.003499`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.003422`
- `lag_02__CT4__molly`: contribution `-0.002987`
- `lag_15__CT2__molly`: contribution `-0.002514`
- `lag_13__T4__molly`: contribution `-0.002297`

### tick `27265`, seconds `48.00`, LSTM delta `+0.0916`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.021471`
- `lag_01__T_place_BALCONY`: contribution `+0.011548`
- `lag_00__kill_diff_last_3s`: contribution `+0.006300`
- `lag_04__T_place_BALCONY`: contribution `+0.006162`
- `lag_05__T_place_BALCONY`: contribution `+0.004285`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.002915`
- `lag_08__CT5__flash_duration`: contribution `+0.002341`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.001889`
