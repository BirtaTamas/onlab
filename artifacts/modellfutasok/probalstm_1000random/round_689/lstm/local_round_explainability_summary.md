# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `5628`, seconds `68.50`, LSTM `0.1161`, delta `-0.2340`
- tick `3420`, seconds `34.00`, LSTM `0.3212`, delta `-0.2221`
- tick `5596`, seconds `68.00`, LSTM `0.3502`, delta `+0.2216`
- tick `3452`, seconds `34.50`, LSTM `0.2052`, delta `-0.1161`
- tick `5756`, seconds `70.50`, LSTM `0.0667`, delta `-0.1150`
- tick `5436`, seconds `65.50`, LSTM `0.0755`, delta `+0.0563`
- tick `5724`, seconds `70.00`, LSTM `0.1817`, delta `+0.0552`
- tick `3516`, seconds `35.50`, LSTM `0.1164`, delta `-0.0444`
- tick `3484`, seconds `35.00`, LSTM `0.1608`, delta `-0.0443`
- tick `5788`, seconds `71.00`, LSTM `0.0238`, delta `-0.0429`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005256`, |coef| `0.005256`
- `lag_00__damage_diff_last_5s`: coefficient `0.004772`, |coef| `0.004772`
- `lag_00__CT_kills_last_3s`: coefficient `0.003599`, |coef| `0.003599`
- `lag_00__T_kills_last_3s`: coefficient `-0.002969`, |coef| `0.002969`
- `lag_11__T4__duck_amount`: coefficient `-0.002730`, |coef| `0.002730`
- `lag_07__CT3__is_walking`: coefficient `0.002707`, |coef| `0.002707`
- `lag_00__CT_damage_last_5s`: coefficient `0.002655`, |coef| `0.002655`
- `lag_14__T3__is_walking`: coefficient `-0.002403`, |coef| `0.002403`
- `lag_00__CT_place_RUINS`: coefficient `0.002354`, |coef| `0.002354`
- `lag_05__T4__duck_amount`: coefficient `0.002296`, |coef| `0.002296`
- `lag_00__CT_velocity_mean`: coefficient `-0.002174`, |coef| `0.002174`
- `lag_15__T5__is_walking`: coefficient `-0.002172`, |coef| `0.002172`
- `lag_00__T_damage_last_5s`: coefficient `-0.002152`, |coef| `0.002152`
- `lag_00__T5__is_walking`: coefficient `0.002125`, |coef| `0.002125`
- `lag_00__T_place_BANANA`: coefficient `-0.002112`, |coef| `0.002112`

## Top 10 utility ridge features

- `lag_15__T4__flash_duration`: coefficient `-0.001204` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001129` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.001065` (lowers CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `-0.001059` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000984` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000977` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.000843` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.000829` (raises CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000809` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.000766` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005256` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004772` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003599` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002969` (lowers CT win probability)
- `lag_11__T4__duck_amount`: coefficient `-0.002730` (lowers CT win probability)
- `lag_07__CT3__is_walking`: coefficient `0.002707` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002655` (raises CT win probability)
- `lag_14__T3__is_walking`: coefficient `-0.002403` (lowers CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.002354` (raises CT win probability)
- `lag_05__T4__duck_amount`: coefficient `0.002296` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `5628`, seconds `68.50`, LSTM delta `-0.2340`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.025302`
- `lag_00__CT_place_RUINS`: contribution `-0.016448`
- `lag_00__damage_diff_last_5s`: contribution `-0.011197`
- `lag_00__CT_kills_last_3s`: contribution `-0.010391`
- `lag_00__T_kills_last_3s`: contribution `-0.009406`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3420`, seconds `34.00`, LSTM delta `-0.2221`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012651`
- `lag_00__damage_diff_last_5s`: contribution `-0.010766`
- `lag_00__T_kills_last_3s`: contribution `-0.009406`
- `lag_07__CT_place_TOPOFMID`: contribution `-0.007494`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.007353`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5596`, seconds `68.00`, LSTM delta `+0.2216`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012651`
- `lag_00__CT_kills_last_3s`: contribution `+0.010391`
- `lag_00__damage_diff_last_5s`: contribution `+0.010013`
- `lag_11__T4__duck_amount`: contribution `+0.008811`
- `lag_05__T4__duck_amount`: contribution `+0.008490`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3452`, seconds `34.50`, LSTM delta `-0.1161`

Top all feature movements:
- `lag_08__CT_place_TOPOFMID`: contribution `-0.007353`
- `lag_09__CT_place_TOPOFMID`: contribution `-0.007036`
- `lag_01__T4__duck_amount`: contribution `-0.006397`
- `lag_02__CT_place_MIDDLE`: contribution `-0.005320`
- `lag_08__CT_place_MIDDLE`: contribution `-0.005075`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5756`, seconds `70.50`, LSTM delta `-0.1150`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.016042`
- `lag_00__kill_diff_last_3s`: contribution `-0.012651`
- `lag_00__T_kills_last_3s`: contribution `-0.009406`
- `lag_04__kill_diff_last_3s`: contribution `-0.007084`
- `lag_04__CT_place_RUINS`: contribution `-0.006920`

Top utility-only movements:
- No utility movement among the top local contributors.
