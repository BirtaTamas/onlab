# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `20`

## Largest probability jumps

- tick `134637`, seconds `46.00`, LSTM `0.6968`, delta `-0.2224`
- tick `134349`, seconds `41.50`, LSTM `0.7270`, delta `+0.1872`
- tick `136365`, seconds `73.00`, LSTM `0.1127`, delta `-0.1812`
- tick `134541`, seconds `44.50`, LSTM `0.9105`, delta `+0.1512`
- tick `135757`, seconds `63.50`, LSTM `0.2553`, delta `+0.1238`
- tick `135053`, seconds `52.50`, LSTM `0.3411`, delta `-0.1075`
- tick `134829`, seconds `49.00`, LSTM `0.6126`, delta `-0.1008`
- tick `134509`, seconds `44.00`, LSTM `0.7593`, delta `+0.0988`
- tick `134381`, seconds `42.00`, LSTM `0.6300`, delta `-0.0969`
- tick `136461`, seconds `74.50`, LSTM `0.1320`, delta `+0.0947`

## Top 15 local ridge features

- `lag_11__CT_place_UPPERPARK`: coefficient `0.003756`, |coef| `0.003756`
- `lag_03__CT_place_LOBBY`: coefficient `0.002870`, |coef| `0.002870`
- `lag_02__CT_place_LOBBY`: coefficient `0.002520`, |coef| `0.002520`
- `lag_00__CT_place_BACKOFA`: coefficient `0.002424`, |coef| `0.002424`
- `lag_00__CT_place_LOBBY`: coefficient `0.002270`, |coef| `0.002270`
- `lag_00__kill_diff_last_3s`: coefficient `0.002186`, |coef| `0.002186`
- `lag_00__damage_diff_last_5s`: coefficient `0.001963`, |coef| `0.001963`
- `lag_04__CT_place_STAIRS`: coefficient `0.001893`, |coef| `0.001893`
- `lag_05__T3__is_scoped`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_00__T4__duck_amount`: coefficient `-0.001861`, |coef| `0.001861`
- `lag_10__CT_place_UPPERPARK`: coefficient `0.001857`, |coef| `0.001857`
- `lag_02__CT_place_UPPERPARK`: coefficient `-0.001843`, |coef| `0.001843`
- `lag_00__CT3__duck_amount`: coefficient `0.001839`, |coef| `0.001839`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001732`, |coef| `0.001732`
- `lag_07__CT_place_STORAGEROOM`: coefficient `-0.001710`, |coef| `0.001710`

## Top 10 utility ridge features

- `lag_00__CT5__flash`: coefficient `0.000855` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000723` (raises CT win probability)
- `lag_11__CT3__smoke`: coefficient `-0.000703` (lowers CT win probability)
- `lag_10__CT4__smoke`: coefficient `0.000689` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000672` (raises CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.000653` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000608` (raises CT win probability)
- `lag_08__CT4__smoke`: coefficient `0.000590` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000554` (lowers CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.000543` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_UPPERPARK`: coefficient `0.003756` (raises CT win probability)
- `lag_03__CT_place_LOBBY`: coefficient `0.002870` (raises CT win probability)
- `lag_02__CT_place_LOBBY`: coefficient `0.002520` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.002424` (raises CT win probability)
- `lag_00__CT_place_LOBBY`: coefficient `0.002270` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002186` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001963` (raises CT win probability)
- `lag_04__CT_place_STAIRS`: coefficient `0.001893` (raises CT win probability)
- `lag_05__T3__is_scoped`: coefficient `-0.001871` (lowers CT win probability)
- `lag_00__T4__duck_amount`: coefficient `-0.001861` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `134637`, seconds `46.00`, LSTM delta `-0.2224`

Top all feature movements:
- `lag_11__CT_place_UPPERPARK`: contribution `-0.026737`
- `lag_03__CT_place_BACKOFA`: contribution `-0.009976`
- `lag_06__T3__is_scoped`: contribution `-0.008103`
- `lag_14__T3__is_scoped`: contribution `-0.007801`
- `lag_00__T4__duck_amount`: contribution `-0.006881`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `134349`, seconds `41.50`, LSTM delta `+0.1872`

Top all feature movements:
- `lag_11__CT_place_UPPERPARK`: contribution `+0.026737`
- `lag_02__CT_place_UPPERPARK`: contribution `+0.013120`
- `lag_05__T3__is_scoped`: contribution `+0.012001`
- `lag_00__CT3__duck_amount`: contribution `+0.006844`
- `lag_00__T4__duck_amount`: contribution `+0.006689`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `136365`, seconds `73.00`, LSTM delta `-0.1812`

Top all feature movements:
- `lag_03__CT_place_LOBBY`: contribution `-0.023491`
- `lag_00__CT_place_BACKOFA`: contribution `-0.023412`
- `lag_02__CT_place_LOBBY`: contribution `-0.020626`
- `lag_00__T_shots_fired_sum`: contribution `-0.006493`
- `lag_00__T4__duck_amount`: contribution `-0.005698`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `134541`, seconds `44.50`, LSTM delta `+0.1512`

Top all feature movements:
- `lag_00__CT_place_BACKOFA`: contribution `+0.023412`
- `lag_04__CT_place_STAIRS`: contribution `+0.014732`
- `lag_00__CT_place_STAIRS`: contribution `-0.007059`
- `lag_03__T3__is_scoped`: contribution `+0.006812`
- `lag_00__CT3__duck_amount`: contribution `+0.006086`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.002485`

### tick `135757`, seconds `63.50`, LSTM delta `+0.1238`

Top all feature movements:
- `lag_07__CT_place_STORAGEROOM`: contribution `+0.036572`
- `lag_00__CT_place_STORAGEROOM`: contribution `+0.026598`
- `lag_00__CT_place_LOBBY`: contribution `+0.018586`
- `lag_07__CT_place_LOBBY`: contribution `+0.005434`
- `lag_15__CT_place_STAIRS`: contribution `+0.005259`

Top utility-only movements:
- No utility movement among the top local contributors.
