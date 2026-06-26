# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `26`

## Largest probability jumps

- tick `215625`, seconds `29.50`, LSTM `0.7550`, delta `+0.2078`
- tick `216745`, seconds `47.00`, LSTM `0.9181`, delta `+0.1893`
- tick `214793`, seconds `16.50`, LSTM `0.5424`, delta `-0.1142`
- tick `214697`, seconds `15.00`, LSTM `0.6052`, delta `+0.0852`
- tick `218281`, seconds `71.00`, LSTM `0.9650`, delta `+0.0714`
- tick `214729`, seconds `15.50`, LSTM `0.6460`, delta `+0.0408`
- tick `215945`, seconds `34.50`, LSTM `0.7856`, delta `-0.0324`
- tick `215817`, seconds `32.50`, LSTM `0.7857`, delta `-0.0320`
- tick `216233`, seconds `39.00`, LSTM `0.7504`, delta `-0.0319`
- tick `215657`, seconds `30.00`, LSTM `0.7846`, delta `+0.0296`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002948`, |coef| `0.002948`
- `lag_00__kill_diff_last_3s`: coefficient `0.002626`, |coef| `0.002626`
- `lag_01__T1__flash_duration`: coefficient `0.002481`, |coef| `0.002481`
- `lag_00__CT_damage_last_5s`: coefficient `0.002399`, |coef| `0.002399`
- `lag_00__damage_diff_last_5s`: coefficient `0.002272`, |coef| `0.002272`
- `lag_00__T_place_MAINHALL`: coefficient `-0.002051`, |coef| `0.002051`
- `lag_00__T_place_RAMP`: coefficient `-0.001979`, |coef| `0.001979`
- `lag_07__CT_place_SIDEENTRANCE`: coefficient `0.001908`, |coef| `0.001908`
- `lag_00__T2__flash`: coefficient `-0.001666`, |coef| `0.001666`
- `lag_00__T_spread_xy`: coefficient `-0.001666`, |coef| `0.001666`
- `lag_15__T1__duck_amount`: coefficient `0.001601`, |coef| `0.001601`
- `lag_02__CT2__duck_amount`: coefficient `-0.001557`, |coef| `0.001557`
- `lag_15__CT_place_TOPOFMID`: coefficient `0.001513`, |coef| `0.001513`
- `lag_11__T_place_TSIDELOWER`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_07__CT1__is_walking`: coefficient `0.001426`, |coef| `0.001426`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `0.002481` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001666` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001243` (lowers CT win probability)
- `lag_11__CT5__smoke`: coefficient `-0.001107` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.001077` (raises CT win probability)
- `lag_07__CT2__smoke`: coefficient `-0.001063` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001004` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000996` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.000994` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000967` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002948` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002626` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002399` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002272` (raises CT win probability)
- `lag_00__T_place_MAINHALL`: coefficient `-0.002051` (lowers CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.001979` (lowers CT win probability)
- `lag_07__CT_place_SIDEENTRANCE`: coefficient `0.001908` (raises CT win probability)
- `lag_00__T_spread_xy`: coefficient `-0.001666` (lowers CT win probability)
- `lag_15__T1__duck_amount`: coefficient `0.001601` (raises CT win probability)
- `lag_02__CT2__duck_amount`: coefficient `-0.001557` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `215625`, seconds `29.50`, LSTM delta `+0.2078`

Top all feature movements:
- `lag_01__T1__flash_duration`: contribution `+0.018235`
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_00__T_place_MAINHALL`: contribution `+0.007403`
- `lag_00__kill_diff_last_3s`: contribution `+0.006321`
- `lag_15__T1__duck_amount`: contribution `+0.006267`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `+0.018235`
- `lag_00__T2__flash`: contribution `+0.004905`
- `lag_09__T1__flash_duration`: contribution `+0.003402`
- `lag_01__T_flash_duration_sum`: contribution `+0.003293`

### tick `216745`, seconds `47.00`, LSTM delta `+0.1893`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_07__CT_place_SIDEENTRANCE`: contribution `+0.007679`
- `lag_00__T_place_RAMP`: contribution `+0.007000`
- `lag_00__kill_diff_last_3s`: contribution `+0.006321`
- `lag_02__CT2__duck_amount`: contribution `+0.005932`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `214793`, seconds `16.50`, LSTM delta `-0.1142`

Top all feature movements:
- `lag_00__T_place_MAINHALL`: contribution `-0.007403`
- `lag_00__kill_diff_last_3s`: contribution `-0.006321`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.005663`
- `lag_15__T3__flash_duration`: contribution `-0.004701`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.004041`

Top utility-only movements:
- `lag_15__T3__flash_duration`: contribution `-0.004701`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.004041`
- `lag_03__T5__flash_duration`: contribution `-0.003243`
- `lag_03__CT5__flash_duration`: contribution `-0.002737`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.002085`

### tick `214697`, seconds `15.00`, LSTM delta `+0.0852`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_00__kill_diff_last_3s`: contribution `+0.006321`
- `lag_15__CT_place_TOPOFMID`: contribution `+0.005492`
- `lag_00__CT_damage_last_5s`: contribution `+0.005126`
- `lag_00__damage_diff_last_5s`: contribution `+0.005023`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.004171`
- `lag_00__CT5__flash_duration`: contribution `+0.001879`
- `lag_09__CT_active_infernos`: contribution `+0.001795`
- `lag_09__active_infernos_total`: contribution `+0.001788`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.001425`

### tick `218281`, seconds `71.00`, LSTM delta `+0.0714`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_00__kill_diff_last_3s`: contribution `+0.006321`
- `lag_00__CT_damage_last_5s`: contribution `+0.005231`
- `lag_00__damage_diff_last_5s`: contribution `+0.005125`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003898`

Top utility-only movements:
- No utility movement among the top local contributors.
