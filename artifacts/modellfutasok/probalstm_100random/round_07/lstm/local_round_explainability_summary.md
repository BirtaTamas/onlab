# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `17`

## Largest probability jumps

- tick `125789`, seconds `72.00`, LSTM `0.2238`, delta `-0.2851`
- tick `121725`, seconds `8.50`, LSTM `0.5825`, delta `-0.2163`
- tick `122173`, seconds `15.50`, LSTM `0.5711`, delta `+0.2017`
- tick `121949`, seconds `12.00`, LSTM `0.3392`, delta `-0.1880`
- tick `126173`, seconds `78.00`, LSTM `0.2070`, delta `-0.1617`
- tick `125469`, seconds `67.00`, LSTM `0.2970`, delta `-0.1437`
- tick `126013`, seconds `75.50`, LSTM `0.3130`, delta `+0.1250`
- tick `122141`, seconds `15.00`, LSTM `0.3694`, delta `+0.1037`
- tick `125533`, seconds `68.00`, LSTM `0.3630`, delta `+0.1033`
- tick `121981`, seconds `12.50`, LSTM `0.2404`, delta `-0.0988`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002943`, |coef| `0.002943`
- `lag_00__T_kills_last_3s`: coefficient `-0.002639`, |coef| `0.002639`
- `lag_07__CT_place_SHOP`: coefficient `-0.002402`, |coef| `0.002402`
- `lag_06__CT1__duck_amount`: coefficient `-0.002251`, |coef| `0.002251`
- `lag_01__CT_place_SHOP`: coefficient `-0.002159`, |coef| `0.002159`
- `lag_14__CT_place_SHOP`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001948`, |coef| `0.001948`
- `lag_07__CT2__duck_amount`: coefficient `0.001819`, |coef| `0.001819`
- `lag_07__CT1__duck_amount`: coefficient `-0.001793`, |coef| `0.001793`
- `lag_00__T_place_TRUCK`: coefficient `-0.001769`, |coef| `0.001769`
- `lag_09__CT_place_JUNGLE`: coefficient `0.001729`, |coef| `0.001729`
- `lag_04__T_place_TRUCK`: coefficient `0.001718`, |coef| `0.001718`
- `lag_00__damage_diff_last_5s`: coefficient `0.001702`, |coef| `0.001702`
- `lag_14__CT1__duck_amount`: coefficient `0.001665`, |coef| `0.001665`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001634`, |coef| `0.001634`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001275` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.001186` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.001163` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001046` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000959` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000928` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000914` (raises CT win probability)
- `lag_02__CT1__smoke`: coefficient `0.000843` (raises CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.000840` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000838` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002943` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002639` (lowers CT win probability)
- `lag_07__CT_place_SHOP`: coefficient `-0.002402` (lowers CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `-0.002251` (lowers CT win probability)
- `lag_01__CT_place_SHOP`: coefficient `-0.002159` (lowers CT win probability)
- `lag_14__CT_place_SHOP`: coefficient `-0.002122` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001948` (raises CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `0.001819` (raises CT win probability)
- `lag_07__CT1__duck_amount`: coefficient `-0.001793` (lowers CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.001769` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `125789`, seconds `72.00`, LSTM delta `-0.2851`

Top all feature movements:
- `lag_04__T_place_TRUCK`: contribution `-0.029832`
- `lag_08__T_place_TRUCK`: contribution `-0.025490`
- `lag_10__T_place_TRUCK`: contribution `-0.024120`
- `lag_09__CT_place_JUNGLE`: contribution `-0.011090`
- `lag_01__CT_place_SHOP`: contribution `-0.010829`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121725`, seconds `8.50`, LSTM delta `-0.2163`

Top all feature movements:
- `lag_07__CT_place_SHOP`: contribution `-0.024095`
- `lag_00__T_kills_last_3s`: contribution `-0.008362`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008121`
- `lag_00__kill_diff_last_3s`: contribution `-0.007083`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.007037`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122173`, seconds `15.50`, LSTM delta `+0.2017`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.011705`
- `lag_06__T_flash_duration_sum`: contribution `+0.010874`
- `lag_03__T_place_TOPOFMID`: contribution `+0.008450`
- `lag_00__kill_diff_last_3s`: contribution `+0.007083`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006767`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `+0.010874`
- `lag_06__T5__flash_duration`: contribution `+0.003247`
- `lag_06__T2__flash_duration`: contribution `+0.003209`
- `lag_06__T3__flash_duration`: contribution `+0.003021`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.002679`

### tick `121949`, seconds `12.00`, LSTM delta `-0.1880`

Top all feature movements:
- `lag_14__CT_place_SHOP`: contribution `-0.021287`
- `lag_00__T_kills_last_3s`: contribution `-0.008362`
- `lag_10__CT_place_SHOP`: contribution `-0.007287`
- `lag_00__kill_diff_last_3s`: contribution `-0.007083`
- `lag_09__CT_place_SHOP`: contribution `-0.006995`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126173`, seconds `78.00`, LSTM delta `-0.1617`

Top all feature movements:
- `lag_07__T_place_TRUCK`: contribution `-0.015049`
- `lag_05__T_bomb_zone_count`: contribution `-0.009297`
- `lag_00__T3__is_scoped`: contribution `-0.007373`
- `lag_07__CT2__duck_amount`: contribution `-0.006932`
- `lag_13__CT_place_SHOP`: contribution `-0.006838`

Top utility-only movements:
- `lag_11__CT1__molly`: contribution `-0.002014`
