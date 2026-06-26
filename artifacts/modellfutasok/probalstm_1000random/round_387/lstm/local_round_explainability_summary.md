# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `24`

## Largest probability jumps

- tick `195671`, seconds `83.00`, LSTM `0.8705`, delta `+0.3695`
- tick `195191`, seconds `75.50`, LSTM `0.7600`, delta `+0.2905`
- tick `192759`, seconds `37.50`, LSTM `0.0848`, delta `-0.2609`
- tick `195607`, seconds `82.00`, LSTM `0.5406`, delta `-0.2339`
- tick `192183`, seconds `28.50`, LSTM `0.3054`, delta `-0.2202`
- tick `194679`, seconds `67.50`, LSTM `0.2613`, delta `+0.2141`
- tick `192215`, seconds `29.00`, LSTM `0.4339`, delta `+0.1286`
- tick `193399`, seconds `47.50`, LSTM `0.1089`, delta `+0.0931`
- tick `195799`, seconds `85.00`, LSTM `0.9411`, delta `+0.0764`
- tick `195063`, seconds `73.50`, LSTM `0.4492`, delta `+0.0719`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006967`, |coef| `0.006967`
- `lag_00__CT_kills_last_3s`: coefficient `0.005164`, |coef| `0.005164`
- `lag_00__damage_diff_last_5s`: coefficient `0.004328`, |coef| `0.004328`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004137`, |coef| `0.004137`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.004113`, |coef| `0.004113`
- `lag_14__T_place_JUNGLE`: coefficient `-0.003957`, |coef| `0.003957`
- `lag_02__CT_place_STAIRS`: coefficient `0.003599`, |coef| `0.003599`
- `lag_06__CT_place_LADDER`: coefficient `-0.003504`, |coef| `0.003504`
- `lag_00__T_kills_last_3s`: coefficient `-0.003504`, |coef| `0.003504`
- `lag_12__T_duck_amount_mean`: coefficient `0.003368`, |coef| `0.003368`
- `lag_12__CT_place_STAIRS`: coefficient `-0.003130`, |coef| `0.003130`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.003106`, |coef| `0.003106`
- `lag_00__T_macro_A`: coefficient `-0.003106`, |coef| `0.003106`
- `lag_14__T4__duck_amount`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_00__alive_diff`: coefficient `0.002739`, |coef| `0.002739`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004137` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001719` (lowers CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.001632` (raises CT win probability)
- `lag_10__T1__molly`: coefficient `-0.001617` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.001576` (raises CT win probability)
- `lag_08__CT4__smoke`: coefficient `-0.001454` (lowers CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001411` (lowers CT win probability)
- `lag_05__CT4__flash`: coefficient `0.001388` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.001382` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.001341` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006967` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005164` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004328` (raises CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.004113` (lowers CT win probability)
- `lag_14__T_place_JUNGLE`: coefficient `-0.003957` (lowers CT win probability)
- `lag_02__CT_place_STAIRS`: coefficient `0.003599` (raises CT win probability)
- `lag_06__CT_place_LADDER`: coefficient `-0.003504` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003504` (lowers CT win probability)
- `lag_12__T_duck_amount_mean`: coefficient `0.003368` (raises CT win probability)
- `lag_12__CT_place_STAIRS`: coefficient `-0.003130` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `195671`, seconds `83.00`, LSTM delta `+0.3695`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025098`
- `lag_12__CT_place_STAIRS`: contribution `+0.024363`
- `lag_12__T_duck_amount_mean`: contribution `+0.019586`
- `lag_00__kill_diff_last_3s`: contribution `+0.016768`
- `lag_00__CT_kills_last_3s`: contribution `+0.014908`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025098`

### tick `195191`, seconds `75.50`, LSTM delta `+0.2905`

Top all feature movements:
- `lag_02__CT_place_STAIRS`: contribution `+0.028012`
- `lag_00__kill_diff_last_3s`: contribution `+0.016768`
- `lag_00__CT_kills_last_3s`: contribution `+0.014908`
- `lag_12__T_duck_amount_mean`: contribution `-0.009793`
- `lag_00__damage_diff_last_5s`: contribution `+0.009763`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `192759`, seconds `37.50`, LSTM delta `-0.2609`

Top all feature movements:
- `lag_14__T_place_JUNGLE`: contribution `-0.051261`
- `lag_06__CT_place_LADDER`: contribution `-0.036437`
- `lag_00__kill_diff_last_3s`: contribution `-0.016768`
- `lag_00__T_kills_last_3s`: contribution `-0.011100`
- `lag_06__CT_place_UNDERPASS`: contribution `-0.010012`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.002762`

### tick `195607`, seconds `82.00`, LSTM delta `-0.2339`

Top all feature movements:
- `lag_10__CT_place_STAIRS`: contribution `-0.020719`
- `lag_12__T_duck_amount_mean`: contribution `-0.019586`
- `lag_15__CT_place_STAIRS`: contribution `-0.019458`
- `lag_00__kill_diff_last_3s`: contribution `-0.016768`
- `lag_00__T_kills_last_3s`: contribution `-0.011100`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `192183`, seconds `28.50`, LSTM delta `-0.2202`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016768`
- `lag_00__T_kills_last_3s`: contribution `-0.011100`
- `lag_04__T_place_CONNECTOR`: contribution `-0.008952`
- `lag_05__T5__flash_duration`: contribution `-0.008205`
- `lag_05__CT_flashed_players`: contribution `-0.007571`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.008205`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.004979`
- `lag_05__CT3__flash_duration`: contribution `-0.003596`
- `lag_00__CT3__flash_duration`: contribution `-0.003536`
