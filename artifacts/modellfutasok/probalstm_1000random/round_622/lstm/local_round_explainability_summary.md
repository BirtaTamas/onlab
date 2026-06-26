# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `9`

## Largest probability jumps

- tick `72143`, seconds `98.50`, LSTM `0.7451`, delta `-0.1987`
- tick `71631`, seconds `90.50`, LSTM `0.8813`, delta `+0.1948`
- tick `70223`, seconds `68.50`, LSTM `0.7148`, delta `+0.1515`
- tick `67855`, seconds `31.50`, LSTM `0.6944`, delta `+0.1083`
- tick `72111`, seconds `98.00`, LSTM `0.9437`, delta `+0.1026`
- tick `72047`, seconds `97.00`, LSTM `0.8147`, delta `-0.0927`
- tick `68207`, seconds `37.00`, LSTM `0.6608`, delta `-0.0768`
- tick `72239`, seconds `100.00`, LSTM `0.8074`, delta `+0.0603`
- tick `68239`, seconds `37.50`, LSTM `0.6068`, delta `-0.0541`
- tick `70703`, seconds `76.00`, LSTM `0.7166`, delta `+0.0476`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002836`, |coef| `0.002836`
- `lag_01__T_place_CONSTRUCTION`: coefficient `0.002733`, |coef| `0.002733`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002292`, |coef| `0.002292`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002071`, |coef| `0.002071`
- `lag_15__CT4__duck_amount`: coefficient `0.001882`, |coef| `0.001882`
- `lag_05__T_place_CONSTRUCTION`: coefficient `0.001858`, |coef| `0.001858`
- `lag_00__T_kills_last_3s`: coefficient `-0.001799`, |coef| `0.001799`
- `lag_00__CT_kills_last_3s`: coefficient `0.001762`, |coef| `0.001762`
- `lag_02__T_place_CONSTRUCTION`: coefficient `0.001652`, |coef| `0.001652`
- `lag_00__damage_diff_last_5s`: coefficient `0.001633`, |coef| `0.001633`
- `lag_12__T_place_PIPE`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_00__CT_place_LOWERPARK`: coefficient `0.001387`, |coef| `0.001387`
- `lag_00__CT4__shots_fired`: coefficient `0.001381`, |coef| `0.001381`
- `lag_09__T_place_PIPE`: coefficient `0.001368`, |coef| `0.001368`
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.001356`, |coef| `0.001356`

## Top 10 utility ridge features

- `lag_02__CT1__flash_duration`: coefficient `0.001133` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000953` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000738` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000680` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000675` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000658` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000593` (lowers CT win probability)
- `lag_07__T4__molly`: coefficient `0.000589` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.000585` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000527` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002836` (raises CT win probability)
- `lag_01__T_place_CONSTRUCTION`: coefficient `0.002733` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002292` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002071` (lowers CT win probability)
- `lag_15__CT4__duck_amount`: coefficient `0.001882` (raises CT win probability)
- `lag_05__T_place_CONSTRUCTION`: coefficient `0.001858` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001799` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001762` (raises CT win probability)
- `lag_02__T_place_CONSTRUCTION`: coefficient `0.001652` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001633` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72143`, seconds `98.50`, LSTM delta `-0.1987`

Top all feature movements:
- `lag_01__T_place_CONSTRUCTION`: contribution `-0.033974`
- `lag_00__T_duck_amount_mean`: contribution `-0.012047`
- `lag_00__CT_shots_fired_sum`: contribution `-0.009554`
- `lag_15__CT4__duck_amount`: contribution `-0.006911`
- `lag_00__kill_diff_last_3s`: contribution `-0.006826`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `-0.002087`

### tick `71631`, seconds `90.50`, LSTM delta `+0.1948`

Top all feature movements:
- `lag_05__T_place_CONSTRUCTION`: contribution `+0.023092`
- `lag_02__T_place_CONSTRUCTION`: contribution `+0.020530`
- `lag_12__T_place_PIPE`: contribution `+0.018319`
- `lag_15__T_place_PIPE`: contribution `+0.015510`
- `lag_00__kill_diff_last_3s`: contribution `+0.006826`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70223`, seconds `68.50`, LSTM delta `+0.1515`

Top all feature movements:
- `lag_15__CT_place_CONSTRUCTION`: contribution `+0.013709`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011147`
- `lag_15__CT_place_LOBBY`: contribution `+0.007694`
- `lag_00__CT_place_CONSTRUCTION`: contribution `+0.007576`
- `lag_00__kill_diff_last_3s`: contribution `+0.006826`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67855`, seconds `31.50`, LSTM delta `+0.1083`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007962`
- `lag_02__CT1__flash_duration`: contribution `+0.007857`
- `lag_00__kill_diff_last_3s`: contribution `+0.006826`
- `lag_00__CT_kills_last_3s`: contribution `+0.005088`
- `lag_02__T2__flash_duration`: contribution `+0.004259`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.007857`
- `lag_02__T2__flash_duration`: contribution `+0.004259`
- `lag_02__T4__flash_duration`: contribution `+0.003259`
- `lag_00__T2__flash_duration`: contribution `+0.002886`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002783`

### tick `72111`, seconds `98.00`, LSTM delta `+0.1026`

Top all feature movements:
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.016856`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007962`
- `lag_15__CT4__duck_amount`: contribution `+0.006911`
- `lag_00__kill_diff_last_3s`: contribution `+0.006826`
- `lag_00__CT_kills_last_3s`: contribution `+0.005088`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.001201`
