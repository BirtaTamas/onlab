# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `12`

## Largest probability jumps

- tick `99459`, seconds `79.50`, LSTM `0.8933`, delta `+0.2203`
- tick `95811`, seconds `22.50`, LSTM `0.4809`, delta `+0.2095`
- tick `95619`, seconds `19.50`, LSTM `0.3341`, delta `-0.1144`
- tick `100099`, seconds `89.50`, LSTM `0.9402`, delta `+0.0994`
- tick `97891`, seconds `55.00`, LSTM `0.5626`, delta `+0.0660`
- tick `99427`, seconds `79.00`, LSTM `0.6730`, delta `-0.0593`
- tick `99139`, seconds `74.50`, LSTM `0.6718`, delta `-0.0473`
- tick `99395`, seconds `78.50`, LSTM `0.7324`, delta `+0.0433`
- tick `95107`, seconds `11.50`, LSTM `0.4407`, delta `+0.0395`
- tick `97475`, seconds `48.50`, LSTM `0.4707`, delta `+0.0383`

## Top 15 local ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `-0.002349`, |coef| `0.002349`
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002093`, |coef| `0.002093`
- `lag_00__kill_diff_last_3s`: coefficient `0.001895`, |coef| `0.001895`
- `lag_04__CT2__flash_duration`: coefficient `0.001567`, |coef| `0.001567`
- `lag_00__CT_kills_last_3s`: coefficient `0.001469`, |coef| `0.001469`
- `lag_03__T4__is_scoped`: coefficient `0.001393`, |coef| `0.001393`
- `lag_04__CT_place_WATER`: coefficient `0.001372`, |coef| `0.001372`
- `lag_00__CT4__is_scoped`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_00__damage_diff_last_5s`: coefficient `0.001243`, |coef| `0.001243`
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.001194`, |coef| `0.001194`
- `lag_04__T_flashed_players`: coefficient `0.001104`, |coef| `0.001104`
- `lag_00__CT_damage_last_5s`: coefficient `0.001073`, |coef| `0.001073`
- `lag_13__T4__is_scoped`: coefficient `-0.001072`, |coef| `0.001072`
- `lag_04__CT_place_BRIDGE`: coefficient `0.001039`, |coef| `0.001039`
- `lag_12__CT_place_LOWERPARK`: coefficient `0.001038`, |coef| `0.001038`

## Top 10 utility ridge features

- `lag_04__CT2__flash_duration`: coefficient `0.001567` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.000727` (raises CT win probability)
- `lag_14__CT5__smoke`: coefficient `-0.000602` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000581` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000574` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000548` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.000542` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000516` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.000485` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `0.000468` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `-0.002349` (lowers CT win probability)
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002093` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001895` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001469` (raises CT win probability)
- `lag_03__T4__is_scoped`: coefficient `0.001393` (raises CT win probability)
- `lag_04__CT_place_WATER`: coefficient `0.001372` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001294` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001243` (raises CT win probability)
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.001194` (lowers CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.001104` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `99459`, seconds `79.50`, LSTM delta `+0.2203`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `+0.026928`
- `lag_04__CT2__flash_duration`: contribution `+0.012515`
- `lag_00__T_place_FOUNTAIN`: contribution `+0.009893`
- `lag_04__CT_place_WATER`: contribution `+0.008340`
- `lag_10__CT_place_BRIDGE`: contribution `+0.007333`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.012515`
- `lag_04__CT_flash_duration_sum`: contribution `+0.002636`

### tick `95811`, seconds `22.50`, LSTM delta `+0.2095`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `+0.026928`
- `lag_13__CT_place_BRIDGE`: contribution `+0.010423`
- `lag_00__T_place_FOUNTAIN`: contribution `+0.009893`
- `lag_00__kill_diff_last_3s`: contribution `+0.009124`
- `lag_10__CT_place_BRIDGE`: contribution `-0.007333`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `95619`, seconds `19.50`, LSTM delta `-0.1144`

Top all feature movements:
- `lag_04__CT_place_BRIDGE`: contribution `-0.011906`
- `lag_01__CT_place_BRIDGE`: contribution `+0.006513`
- `lag_06__T_shots_fired_sum`: contribution `-0.004663`
- `lag_00__kill_diff_last_3s`: contribution `-0.004562`
- `lag_05__T_shots_fired_sum`: contribution `-0.004558`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100099`, seconds `89.50`, LSTM delta `+0.0994`

Top all feature movements:
- `lag_06__CT_flashed_players`: contribution `+0.007010`
- `lag_00__kill_diff_last_3s`: contribution `+0.004562`
- `lag_00__CT4__is_scoped`: contribution `+0.004409`
- `lag_00__CT_kills_last_3s`: contribution `+0.004243`
- `lag_09__CT_place_WALKWAY`: contribution `+0.003746`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.003373`
- `lag_04__CT2__flash_duration`: contribution `-0.001886`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001728`
- `lag_06__T4__flash_duration`: contribution `+0.001592`

### tick `97891`, seconds `55.00`, LSTM delta `+0.0660`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004562`
- `lag_00__CT4__is_scoped`: contribution `+0.004409`
- `lag_00__CT_kills_last_3s`: contribution `+0.004243`
- `lag_09__CT_place_LOWERPARK`: contribution `+0.003543`
- `lag_13__CT_place_WALKWAY`: contribution `+0.002898`

Top utility-only movements:
- No utility movement among the top local contributors.
