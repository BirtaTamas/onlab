# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `110159`, seconds `86.00`, LSTM `0.6284`, delta `+0.2515`
- tick `110095`, seconds `85.00`, LSTM `0.4378`, delta `+0.2410`
- tick `109423`, seconds `74.50`, LSTM `0.2526`, delta `+0.1842`
- tick `109967`, seconds `83.00`, LSTM `0.3432`, delta `-0.1618`
- tick `109807`, seconds `80.50`, LSTM `0.5338`, delta `+0.1555`
- tick `107375`, seconds `42.50`, LSTM `0.2322`, delta `-0.1189`
- tick `107343`, seconds `42.00`, LSTM `0.3511`, delta `-0.1044`
- tick `107535`, seconds `45.00`, LSTM `0.0453`, delta `-0.0959`
- tick `106351`, seconds `26.50`, LSTM `0.5173`, delta `-0.0933`
- tick `107151`, seconds `39.00`, LSTM `0.4275`, delta `-0.0823`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005022`, |coef| `0.005022`
- `lag_00__kill_diff_last_3s`: coefficient `0.003572`, |coef| `0.003572`
- `lag_02__T_flash_alpha_mean`: coefficient `-0.003147`, |coef| `0.003147`
- `lag_01__CT_defusing_count`: coefficient `0.002999`, |coef| `0.002999`
- `lag_14__CT_place_DECON`: coefficient `0.002744`, |coef| `0.002744`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002663`, |coef| `0.002663`
- `lag_00__T_kills_last_3s`: coefficient `-0.002605`, |coef| `0.002605`
- `lag_07__CT_place_DECON`: coefficient `0.002588`, |coef| `0.002588`
- `lag_05__T_place_OBSERVATION`: coefficient `-0.002267`, |coef| `0.002267`
- `lag_00__damage_diff_last_5s`: coefficient `0.002073`, |coef| `0.002073`
- `lag_00__CT_place_SECRET`: coefficient `-0.002048`, |coef| `0.002048`
- `lag_00__T_velocity_mean`: coefficient `-0.001915`, |coef| `0.001915`
- `lag_00__CT_kills_last_3s`: coefficient `0.001911`, |coef| `0.001911`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_01__CT_place_DECON`: coefficient `-0.001843`, |coef| `0.001843`

## Top 10 utility ridge features

- `lag_02__T_flash_alpha_mean`: coefficient `-0.003147` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002663` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001877` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001566` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.001138` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001121` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001083` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001056` (lowers CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.000865` (lowers CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `0.000715` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005022` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003572` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.002999` (raises CT win probability)
- `lag_14__CT_place_DECON`: coefficient `0.002744` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002605` (lowers CT win probability)
- `lag_07__CT_place_DECON`: coefficient `0.002588` (raises CT win probability)
- `lag_05__T_place_OBSERVATION`: coefficient `-0.002267` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002073` (raises CT win probability)
- `lag_00__CT_place_SECRET`: coefficient `-0.002048` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.001915` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `110159`, seconds `86.00`, LSTM delta `+0.2515`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.048681`
- `lag_13__CT_place_DECON`: contribution `+0.022025`
- `lag_02__T_flash_alpha_mean`: contribution `+0.019093`
- `lag_00__kill_diff_last_3s`: contribution `+0.008598`
- `lag_00__T_kills_last_3s`: contribution `+0.008252`

Top utility-only movements:
- `lag_02__T_flash_alpha_mean`: contribution `+0.019093`

### tick `110095`, seconds `85.00`, LSTM delta `+0.2410`

Top all feature movements:
- `lag_14__CT_place_DECON`: contribution `+0.043630`
- `lag_00__T_flash_alpha_mean`: contribution `+0.016160`
- `lag_15__CT_place_DECON`: contribution `+0.013469`
- `lag_11__CT_place_DECON`: contribution `+0.012293`
- `lag_00__kill_diff_last_3s`: contribution `+0.008598`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016160`

### tick `109423`, seconds `74.50`, LSTM delta `+0.1842`

Top all feature movements:
- `lag_05__T_place_OBSERVATION`: contribution `+0.038395`
- `lag_12__CT_place_DECON`: contribution `+0.019871`
- `lag_00__kill_diff_last_3s`: contribution `+0.008598`
- `lag_00__CT_kills_last_3s`: contribution `+0.005518`
- `lag_00__damage_diff_last_5s`: contribution `+0.004631`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109967`, seconds `83.00`, LSTM delta `-0.1618`

Top all feature movements:
- `lag_07__CT_place_DECON`: contribution `-0.041159`
- `lag_10__CT_place_DECON`: contribution `-0.028165`
- `lag_11__CT_place_DECON`: contribution `+0.012293`
- `lag_00__kill_diff_last_3s`: contribution `-0.008598`
- `lag_00__T_kills_last_3s`: contribution `-0.008252`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `109807`, seconds `80.50`, LSTM delta `+0.1555`

Top all feature movements:
- `lag_05__CT_place_DECON`: contribution `+0.027811`
- `lag_02__CT_place_DECON`: contribution `+0.024811`
- `lag_00__kill_diff_last_3s`: contribution `+0.008598`
- `lag_00__CT_kills_last_3s`: contribution `+0.005518`
- `lag_10__T5__is_walking`: contribution `+0.004228`

Top utility-only movements:
- No utility movement among the top local contributors.
