# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `28146`, seconds `21.50`, LSTM `0.4013`, delta `+0.2265`
- tick `27986`, seconds `19.00`, LSTM `0.2423`, delta `-0.2209`
- tick `28690`, seconds `30.00`, LSTM `0.1308`, delta `-0.2019`
- tick `28178`, seconds `22.00`, LSTM `0.2051`, delta `-0.1963`
- tick `29394`, seconds `41.00`, LSTM `0.1081`, delta `+0.0696`
- tick `29746`, seconds `46.50`, LSTM `0.0098`, delta `-0.0662`
- tick `29714`, seconds `46.00`, LSTM `0.0760`, delta `-0.0657`
- tick `28018`, seconds `19.50`, LSTM `0.1885`, delta `-0.0538`
- tick `28402`, seconds `25.50`, LSTM `0.2548`, delta `+0.0462`
- tick `28658`, seconds `29.50`, LSTM `0.3327`, delta `+0.0445`

## Top 15 local ridge features

- `lag_01__CT_place_LADDER`: coefficient `-0.002541`, |coef| `0.002541`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002340`, |coef| `0.002340`
- `lag_00__kill_diff_last_3s`: coefficient `0.002042`, |coef| `0.002042`
- `lag_05__T_place_UNDERPASS`: coefficient `-0.002003`, |coef| `0.002003`
- `lag_03__T1__shots_fired`: coefficient `-0.001946`, |coef| `0.001946`
- `lag_03__T_shots_fired_sum`: coefficient `-0.001902`, |coef| `0.001902`
- `lag_15__T_shots_fired_sum`: coefficient `0.001832`, |coef| `0.001832`
- `lag_00__T_kills_last_3s`: coefficient `-0.001789`, |coef| `0.001789`
- `lag_04__T1__shots_fired`: coefficient `0.001743`, |coef| `0.001743`
- `lag_00__damage_diff_last_5s`: coefficient `0.001643`, |coef| `0.001643`
- `lag_15__CT5__is_walking`: coefficient `-0.001509`, |coef| `0.001509`
- `lag_13__CT1__flash_duration`: coefficient `0.001449`, |coef| `0.001449`
- `lag_01__T_flashes_last_5s`: coefficient `0.001429`, |coef| `0.001429`
- `lag_05__CT2__flash_duration`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_00__CT5__duck_amount`: coefficient `0.001276`, |coef| `0.001276`

## Top 10 utility ridge features

- `lag_13__CT1__flash_duration`: coefficient `0.001449` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `0.001429` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001329` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.001206` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.001060` (raises CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001025` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.001024` (raises CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `-0.000970` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `0.000957` (raises CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.000940` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_LADDER`: coefficient `-0.002541` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002340` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002042` (raises CT win probability)
- `lag_05__T_place_UNDERPASS`: coefficient `-0.002003` (lowers CT win probability)
- `lag_03__T1__shots_fired`: coefficient `-0.001946` (lowers CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.001902` (lowers CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `0.001832` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001789` (lowers CT win probability)
- `lag_04__T1__shots_fired`: coefficient `0.001743` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001643` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28146`, seconds `21.50`, LSTM delta `+0.2265`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.018538`
- `lag_03__T1__shots_fired`: contribution `+0.015122`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013008`
- `lag_05__CT2__flash_duration`: contribution `+0.010146`
- `lag_07__T1__flash_duration`: contribution `+0.007330`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.010146`
- `lag_07__T1__flash_duration`: contribution `+0.007330`
- `lag_09__CT1__flash_duration`: contribution `+0.003800`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.003228`

### tick `27986`, seconds `19.00`, LSTM delta `-0.2209`

Top all feature movements:
- `lag_15__T_shots_fired_sum`: contribution `-0.026094`
- `lag_15__T2__shots_fired`: contribution `-0.009814`
- `lag_00__T_shots_fired_sum`: contribution `-0.008142`
- `lag_14__CT2__flash_duration`: contribution `-0.007825`
- `lag_00__CT2__flash_duration`: contribution `-0.006656`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `-0.007825`
- `lag_00__CT2__flash_duration`: contribution `-0.006656`
- `lag_14__T1__flash_duration`: contribution `-0.005592`
- `lag_04__CT1__flash_duration`: contribution `-0.004817`
- `lag_02__T1__flash_duration`: contribution `-0.004519`

### tick `28690`, seconds `30.00`, LSTM delta `-0.2019`

Top all feature movements:
- `lag_01__CT_place_LADDER`: contribution `-0.026419`
- `lag_05__T_place_UNDERPASS`: contribution `-0.007845`
- `lag_13__CT1__flash_duration`: contribution `-0.007126`
- `lag_00__T_kills_last_3s`: contribution `-0.005669`
- `lag_15__T_shots_fired_sum`: contribution `-0.005493`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.007126`
- `lag_09__CT_utility_damage_last_5s`: contribution `-0.003418`
- `lag_15__T_utility_damage_last_5s`: contribution `-0.002909`

### tick `28178`, seconds `22.00`, LSTM delta `-0.1963`

Top all feature movements:
- `lag_04__T1__shots_fired`: contribution `-0.013540`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013008`
- `lag_04__T_shots_fired_sum`: contribution `-0.008781`
- `lag_06__CT2__flash_duration`: contribution `-0.007817`
- `lag_05__T5__shots_fired`: contribution `-0.006073`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `-0.007817`
- `lag_10__CT1__flash_duration`: contribution `-0.004963`
- `lag_08__T1__flash_duration`: contribution `-0.004836`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.002262`

### tick `29394`, seconds `41.00`, LSTM delta `+0.0696`

Top all feature movements:
- `lag_01__T_flashes_last_5s`: contribution `+0.012946`
- `lag_05__T_place_UNDERPASS`: contribution `+0.007845`
- `lag_00__kill_diff_last_3s`: contribution `+0.004915`
- `lag_00__damage_diff_last_5s`: contribution `+0.003707`
- `lag_06__CT4__duck_amount`: contribution `+0.002421`

Top utility-only movements:
- `lag_01__T_flashes_last_5s`: contribution `+0.012946`
- `lag_15__CT_active_smokes`: contribution `+0.001264`
