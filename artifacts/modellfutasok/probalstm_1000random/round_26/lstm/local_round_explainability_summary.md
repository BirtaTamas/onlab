# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `35778`, seconds `106.00`, LSTM `0.4612`, delta `+0.3986`
- tick `36034`, seconds `110.00`, LSTM `0.7442`, delta `+0.3837`
- tick `34146`, seconds `80.50`, LSTM `0.4484`, delta `+0.3756`
- tick `34178`, seconds `81.00`, LSTM `0.0895`, delta `-0.3589`
- tick `33922`, seconds `77.00`, LSTM `0.3853`, delta `+0.2790`
- tick `34082`, seconds `79.50`, LSTM `0.1200`, delta `-0.1521`
- tick `35810`, seconds `106.50`, LSTM `0.3393`, delta `-0.1219`
- tick `33794`, seconds `75.00`, LSTM `0.1735`, delta `+0.1135`
- tick `34018`, seconds `78.50`, LSTM `0.2584`, delta `-0.1086`
- tick `36162`, seconds `112.00`, LSTM `0.9381`, delta `+0.0759`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.005074`, |coef| `0.005074`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005014`, |coef| `0.005014`
- `lag_00__kill_diff_last_3s`: coefficient `0.004500`, |coef| `0.004500`
- `lag_15__T1__flash_duration`: coefficient `-0.004430`, |coef| `0.004430`
- `lag_00__damage_diff_last_5s`: coefficient `0.004062`, |coef| `0.004062`
- `lag_04__T_duck_amount_mean`: coefficient `-0.003983`, |coef| `0.003983`
- `lag_00__CT_damage_last_5s`: coefficient `0.003913`, |coef| `0.003913`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003829`, |coef| `0.003829`
- `lag_07__T_place_BDOORS`: coefficient `0.003633`, |coef| `0.003633`
- `lag_00__CT4__duck_amount`: coefficient `0.003604`, |coef| `0.003604`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003448`, |coef| `0.003448`
- `lag_14__CT1__duck_amount`: coefficient `0.003368`, |coef| `0.003368`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003346`, |coef| `0.003346`
- `lag_00__T_macro_B`: coefficient `-0.003346`, |coef| `0.003346`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002895`, |coef| `0.002895`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005014` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.004430` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.002827` (lowers CT win probability)
- `lag_14__T1__molly`: coefficient `-0.002564` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.002556` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.002117` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.001888` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `-0.001826` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.001750` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.001594` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.005074` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004500` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004062` (raises CT win probability)
- `lag_04__T_duck_amount_mean`: coefficient `-0.003983` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003913` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003829` (raises CT win probability)
- `lag_07__T_place_BDOORS`: coefficient `0.003633` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.003604` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003448` (raises CT win probability)
- `lag_14__CT1__duck_amount`: coefficient `0.003368` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35778`, seconds `106.00`, LSTM delta `+0.3986`

Top all feature movements:
- `lag_15__T1__flash_duration`: contribution `+0.019512`
- `lag_02__T1__is_scoped`: contribution `+0.015996`
- `lag_10__T1__is_scoped`: contribution `+0.015190`
- `lag_00__CT_kills_last_3s`: contribution `+0.014648`
- `lag_05__T1__is_scoped`: contribution `+0.014528`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.019512`
- `lag_12__T_B_site_active_infernos`: contribution `+0.007227`

### tick `36034`, seconds `110.00`, LSTM delta `+0.3837`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.030422`
- `lag_04__T_duck_amount_mean`: contribution `+0.023165`
- `lag_10__T1__is_scoped`: contribution `+0.015190`
- `lag_00__CT_kills_last_3s`: contribution `+0.014648`
- `lag_00__kill_diff_last_3s`: contribution `+0.010830`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.030422`
- `lag_06__T_B_site_active_infernos`: contribution `+0.005338`

### tick `34146`, seconds `80.50`, LSTM delta `+0.3756`

Top all feature movements:
- `lag_02__T_place_HOLE`: contribution `+0.047042`
- `lag_07__T_place_BDOORS`: contribution `+0.045446`
- `lag_14__T_place_BDOORS`: contribution `+0.031165`
- `lag_11__T_place_BDOORS`: contribution `+0.020237`
- `lag_00__damage_diff_last_5s`: contribution `+0.017594`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34178`, seconds `81.00`, LSTM delta `-0.3589`

Top all feature movements:
- `lag_15__T_place_BDOORS`: contribution `-0.071908`
- `lag_03__T_place_HOLE`: contribution `-0.051809`
- `lag_05__T_place_HOLE`: contribution `-0.051432`
- `lag_04__CT_place_OUTSIDELONG`: contribution `-0.027315`
- `lag_01__CT_place_OUTSIDELONG`: contribution `-0.024236`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33922`, seconds `77.00`, LSTM delta `+0.2790`

Top all feature movements:
- `lag_07__T_place_BDOORS`: contribution `+0.090892`
- `lag_00__T_place_MIDDOORS`: contribution `+0.019287`
- `lag_00__damage_diff_last_5s`: contribution `+0.015670`
- `lag_00__CT_kills_last_3s`: contribution `+0.014648`
- `lag_00__CT_damage_last_5s`: contribution `+0.014587`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.004371`
