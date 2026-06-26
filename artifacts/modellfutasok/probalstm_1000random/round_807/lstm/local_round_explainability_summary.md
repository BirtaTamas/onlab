# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `19`

## Largest probability jumps

- tick `162271`, seconds `46.50`, LSTM `0.1797`, delta `-0.3284`
- tick `165599`, seconds `98.50`, LSTM `0.8815`, delta `+0.2705`
- tick `162335`, seconds `47.50`, LSTM `0.3450`, delta `+0.2379`
- tick `160223`, seconds `14.50`, LSTM `0.3931`, delta `+0.1265`
- tick `165119`, seconds `91.00`, LSTM `0.5071`, delta `+0.0956`
- tick `162431`, seconds `49.00`, LSTM `0.2716`, delta `-0.0802`
- tick `162303`, seconds `47.00`, LSTM `0.1071`, delta `-0.0726`
- tick `159327`, seconds `0.50`, LSTM `0.1253`, delta `-0.0703`
- tick `161439`, seconds `33.50`, LSTM `0.4561`, delta `-0.0578`
- tick `165375`, seconds `95.00`, LSTM `0.5164`, delta `+0.0566`

## Top 15 local ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003037`, |coef| `0.003037`
- `lag_01__CT_flashed_players`: coefficient `-0.002676`, |coef| `0.002676`
- `lag_00__kill_diff_last_3s`: coefficient `0.002594`, |coef| `0.002594`
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.002586`, |coef| `0.002586`
- `lag_00__damage_diff_last_5s`: coefficient `0.002453`, |coef| `0.002453`
- `lag_01__CT4__flash_duration`: coefficient `-0.002415`, |coef| `0.002415`
- `lag_07__CT5__duck_amount`: coefficient `-0.002192`, |coef| `0.002192`
- `lag_00__CT_kills_last_3s`: coefficient `0.002121`, |coef| `0.002121`
- `lag_11__T_bomb_zone_count`: coefficient `0.001981`, |coef| `0.001981`
- `lag_02__CT1__duck_amount`: coefficient `0.001980`, |coef| `0.001980`
- `lag_14__CT2__shots_fired`: coefficient `-0.001962`, |coef| `0.001962`
- `lag_13__CT1__duck_amount`: coefficient `-0.001827`, |coef| `0.001827`
- `lag_14__CT_shots_fired_sum`: coefficient `-0.001805`, |coef| `0.001805`
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.001781`, |coef| `0.001781`
- `lag_13__T_place_RAMP`: coefficient `-0.001689`, |coef| `0.001689`

## Top 10 utility ridge features

- `lag_01__CT4__flash_duration`: coefficient `-0.002415` (lowers CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.001638` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001114` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.001043` (raises CT win probability)
- `lag_07__T2__molly`: coefficient `0.000970` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000885` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.000843` (lowers CT win probability)
- `lag_04__CT2__flash`: coefficient `0.000826` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000769` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `0.000755` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003037` (lowers CT win probability)
- `lag_01__CT_flashed_players`: coefficient `-0.002676` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002594` (raises CT win probability)
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.002586` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002453` (raises CT win probability)
- `lag_07__CT5__duck_amount`: coefficient `-0.002192` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002121` (raises CT win probability)
- `lag_11__T_bomb_zone_count`: coefficient `0.001981` (raises CT win probability)
- `lag_02__CT1__duck_amount`: coefficient `0.001980` (raises CT win probability)
- `lag_14__CT2__shots_fired`: coefficient `-0.001962` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `162271`, seconds `46.50`, LSTM delta `-0.3284`

Top all feature movements:
- `lag_01__CT_flashed_players`: contribution `-0.017584`
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.014819`
- `lag_01__CT4__flash_duration`: contribution `-0.014387`
- `lag_07__CT5__duck_amount`: contribution `-0.008275`
- `lag_02__CT1__duck_amount`: contribution `-0.007556`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.014387`
- `lag_01__CT_flash_duration_sum`: contribution `-0.006663`

### tick `165599`, seconds `98.50`, LSTM delta `+0.2705`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `+0.026331`
- `lag_14__CT2__shots_fired`: contribution `+0.020481`
- `lag_02__CT_place_TSIDEUPPER`: contribution `+0.019440`
- `lag_13__T_place_RAMP`: contribution `+0.011946`
- `lag_11__T_bomb_zone_count`: contribution `+0.011535`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `162335`, seconds `47.50`, LSTM delta `+0.2379`

Top all feature movements:
- `lag_07__CT5__duck_amount`: contribution `+0.008275`
- `lag_03__CT_flashed_players`: contribution `+0.007532`
- `lag_03__CT4__flash_duration`: contribution `+0.006639`
- `lag_00__kill_diff_last_3s`: contribution `+0.006243`
- `lag_00__CT_kills_last_3s`: contribution `+0.006122`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `+0.006639`

### tick `160223`, seconds `14.50`, LSTM delta `+0.1265`

Top all feature movements:
- `lag_02__CT1__duck_amount`: contribution `+0.007556`
- `lag_00__kill_diff_last_3s`: contribution `+0.006243`
- `lag_00__CT_kills_last_3s`: contribution `+0.006122`
- `lag_00__damage_diff_last_5s`: contribution `+0.003763`
- `lag_02__CT3__is_scoped`: contribution `+0.003315`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `+0.002948`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.001724`

### tick `165119`, seconds `91.00`, LSTM delta `+0.0956`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.014819`
- `lag_00__CT_kills_last_3s`: contribution `+0.006122`
- `lag_12__T_shots_fired_sum`: contribution `+0.005633`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003653`
- `lag_12__T5__shots_fired`: contribution `+0.003573`

Top utility-only movements:
- No utility movement among the top local contributors.
