# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `47418`, seconds `30.50`, LSTM `0.7411`, delta `+0.1966`
- tick `47866`, seconds `37.50`, LSTM `0.8583`, delta `+0.1759`
- tick `46778`, seconds `20.50`, LSTM `0.4803`, delta `-0.1162`
- tick `48218`, seconds `43.00`, LSTM `0.9492`, delta `+0.1053`
- tick `47258`, seconds `28.00`, LSTM `0.5545`, delta `+0.0788`
- tick `47514`, seconds `32.00`, LSTM `0.7690`, delta `+0.0584`
- tick `47226`, seconds `27.50`, LSTM `0.4757`, delta `+0.0550`
- tick `47610`, seconds `33.50`, LSTM `0.7552`, delta `-0.0544`
- tick `47546`, seconds `32.50`, LSTM `0.8184`, delta `+0.0495`
- tick `46746`, seconds `20.00`, LSTM `0.5965`, delta `-0.0378`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002199`, |coef| `0.002199`
- `lag_00__CT_kills_last_3s`: coefficient `0.002005`, |coef| `0.002005`
- `lag_02__T_flashed_players`: coefficient `-0.001702`, |coef| `0.001702`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001678`, |coef| `0.001678`
- `lag_06__T5__flash_duration`: coefficient `-0.001653`, |coef| `0.001653`
- `lag_07__T_place_HOUSE`: coefficient `0.001629`, |coef| `0.001629`
- `lag_00__T_place_HOUSE`: coefficient `-0.001589`, |coef| `0.001589`
- `lag_14__T_place_SIDEHALL`: coefficient `0.001461`, |coef| `0.001461`
- `lag_00__damage_diff_last_5s`: coefficient `0.001392`, |coef| `0.001392`
- `lag_14__T_place_MIDDLE`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001337`, |coef| `0.001337`
- `lag_03__CT_place_HOUSE`: coefficient `0.001194`, |coef| `0.001194`
- `lag_03__T_flashed_players`: coefficient `-0.001189`, |coef| `0.001189`
- `lag_14__T_flashed_players`: coefficient `-0.001188`, |coef| `0.001188`
- `lag_00__CT_damage_last_5s`: coefficient `0.001125`, |coef| `0.001125`

## Top 10 utility ridge features

- `lag_06__T5__flash_duration`: coefficient `-0.001653` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000927` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000911` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `0.000848` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.000805` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000780` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `-0.000771` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000771` (lowers CT win probability)
- `lag_15__T3__flash_duration`: coefficient `-0.000766` (lowers CT win probability)
- `lag_14__CT1__molly`: coefficient `-0.000696` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002199` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002005` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.001702` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001678` (raises CT win probability)
- `lag_07__T_place_HOUSE`: coefficient `0.001629` (raises CT win probability)
- `lag_00__T_place_HOUSE`: coefficient `-0.001589` (lowers CT win probability)
- `lag_14__T_place_SIDEHALL`: coefficient `0.001461` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001392` (raises CT win probability)
- `lag_14__T_place_MIDDLE`: coefficient `-0.001342` (lowers CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001337` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `47418`, seconds `30.50`, LSTM delta `+0.1966`

Top all feature movements:
- `lag_06__T5__flash_duration`: contribution `+0.010504`
- `lag_14__T_place_SIDEHALL`: contribution `+0.009467`
- `lag_07__T_place_HOUSE`: contribution `+0.007161`
- `lag_00__T_place_HOUSE`: contribution `+0.006988`
- `lag_00__CT_kills_last_3s`: contribution `+0.005788`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.010504`
- `lag_06__T_flash_duration_sum`: contribution `+0.003970`
- `lag_06__T3__flash_duration`: contribution `+0.002824`

### tick `47866`, seconds `37.50`, LSTM delta `+0.1759`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `+0.008357`
- `lag_00__T_place_HOUSE`: contribution `+0.006988`
- `lag_00__CT_kills_last_3s`: contribution `+0.005788`
- `lag_00__kill_diff_last_3s`: contribution `+0.005294`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004662`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `+0.002360`

### tick `46778`, seconds `20.50`, LSTM delta `-0.1162`

Top all feature movements:
- `lag_02__T_flashed_players`: contribution `-0.016423`
- `lag_00__kill_diff_last_3s`: contribution `-0.005294`
- `lag_14__T_flashed_players`: contribution `-0.004586`
- `lag_05__T3__flash_duration`: contribution `-0.003895`
- `lag_02__T_place_MIDDLE`: contribution `-0.003552`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.003895`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.003130`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.002914`
- `lag_02__T_flash_duration_sum`: contribution `-0.002223`

### tick `48218`, seconds `43.00`, LSTM delta `+0.1053`

Top all feature movements:
- `lag_07__T_place_HOUSE`: contribution `+0.007161`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005828`
- `lag_00__CT_kills_last_3s`: contribution `+0.005788`
- `lag_00__kill_diff_last_3s`: contribution `+0.005294`
- `lag_03__CT_place_HOUSE`: contribution `+0.004217`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47258`, seconds `28.00`, LSTM delta `+0.0788`

Top all feature movements:
- `lag_00__T_place_HOUSE`: contribution `-0.006988`
- `lag_00__CT_kills_last_3s`: contribution `+0.005788`
- `lag_06__T_place_SIDEHALL`: contribution `+0.005604`
- `lag_00__kill_diff_last_3s`: contribution `+0.005294`
- `lag_01__T5__flash_duration`: contribution `+0.005114`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `+0.005114`
- `lag_01__T_flash_duration_sum`: contribution `+0.002199`
- `lag_12__T5__flash_duration`: contribution `+0.001512`
