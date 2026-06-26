# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `2`

## Largest probability jumps

- tick `17922`, seconds `93.00`, LSTM `0.5487`, delta `+0.3741`
- tick `18050`, seconds `95.00`, LSTM `0.6730`, delta `+0.3693`
- tick `17666`, seconds `89.00`, LSTM `0.3004`, delta `-0.2427`
- tick `17986`, seconds `94.00`, LSTM `0.2786`, delta `-0.2234`
- tick `18114`, seconds `96.00`, LSTM `0.8411`, delta `+0.2123`
- tick `16258`, seconds `67.00`, LSTM `0.5693`, delta `-0.2095`
- tick `15522`, seconds `55.50`, LSTM `0.8331`, delta `+0.2090`
- tick `17634`, seconds `88.50`, LSTM `0.5432`, delta `-0.1911`
- tick `17570`, seconds `87.50`, LSTM `0.7068`, delta `+0.1513`
- tick `15234`, seconds `51.00`, LSTM `0.6248`, delta `+0.0875`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.005579`, |coef| `0.005579`
- `lag_00__kill_diff_last_3s`: coefficient `0.004238`, |coef| `0.004238`
- `lag_08__CT_place_TSIDELOWER`: coefficient `-0.003999`, |coef| `0.003999`
- `lag_00__CT_kills_last_3s`: coefficient `0.002998`, |coef| `0.002998`
- `lag_10__CT4__flash_duration`: coefficient `-0.002666`, |coef| `0.002666`
- `lag_11__CT_place_TSIDELOWER`: coefficient `-0.002609`, |coef| `0.002609`
- `lag_09__CT3__flash_duration`: coefficient `0.002573`, |coef| `0.002573`
- `lag_14__CT_place_TSIDELOWER`: coefficient `-0.002416`, |coef| `0.002416`
- `lag_07__T4__flash_duration`: coefficient `-0.002380`, |coef| `0.002380`
- `lag_00__T_kills_last_3s`: coefficient `-0.002288`, |coef| `0.002288`
- `lag_00__CT_defusing_count`: coefficient `0.002275`, |coef| `0.002275`
- `lag_10__CT_place_TSIDELOWER`: coefficient `0.002262`, |coef| `0.002262`
- `lag_08__CT3__is_scoped`: coefficient `0.002196`, |coef| `0.002196`
- `lag_04__CT4__flash_duration`: coefficient `-0.002176`, |coef| `0.002176`
- `lag_00__damage_diff_last_5s`: coefficient `0.002065`, |coef| `0.002065`

## Top 10 utility ridge features

- `lag_10__CT4__flash_duration`: coefficient `-0.002666` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.002573` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.002380` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.002176` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001842` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.001810` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.001483` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001426` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001317` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.001230` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.005579` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004238` (raises CT win probability)
- `lag_08__CT_place_TSIDELOWER`: coefficient `-0.003999` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002998` (raises CT win probability)
- `lag_11__CT_place_TSIDELOWER`: coefficient `-0.002609` (lowers CT win probability)
- `lag_14__CT_place_TSIDELOWER`: coefficient `-0.002416` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002288` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002275` (raises CT win probability)
- `lag_10__CT_place_TSIDELOWER`: coefficient `0.002262` (raises CT win probability)
- `lag_08__CT3__is_scoped`: coefficient `0.002196` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17922`, seconds `93.00`, LSTM delta `+0.3741`

Top all feature movements:
- `lag_08__CT_place_TSIDELOWER`: contribution `+0.054328`
- `lag_04__CT4__flash_duration`: contribution `+0.015372`
- `lag_07__T4__flash_duration`: contribution `+0.014188`
- `lag_00__kill_diff_last_3s`: contribution `+0.010201`
- `lag_08__CT3__is_scoped`: contribution `+0.009988`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.015372`
- `lag_07__T4__flash_duration`: contribution `+0.014188`
- `lag_09__CT1__flash_duration`: contribution `+0.008920`

### tick `18050`, seconds `95.00`, LSTM delta `+0.3693`

Top all feature movements:
- `lag_12__CT_place_TSIDELOWER`: contribution `+0.075790`
- `lag_00__T_flash_alpha_mean`: contribution `+0.011173`
- `lag_00__kill_diff_last_3s`: contribution `+0.010201`
- `lag_00__CT_kills_last_3s`: contribution `+0.008656`
- `lag_11__T4__flash_duration`: contribution `+0.008498`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.011173`
- `lag_11__T4__flash_duration`: contribution `+0.008498`
- `lag_13__CT1__flash_duration`: contribution `+0.005698`
- `lag_08__CT4__flash_duration`: contribution `+0.003594`

### tick `17666`, seconds `89.00`, LSTM delta `-0.2427`

Top all feature movements:
- `lag_12__CT_place_TSIDELOWER`: contribution `-0.075790`
- `lag_11__CT4__flash_duration`: contribution `-0.012178`
- `lag_00__kill_diff_last_3s`: contribution `-0.010201`
- `lag_00__T_kills_last_3s`: contribution `-0.007250`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.006939`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `-0.012178`
- `lag_01__CT1__flash_duration`: contribution `-0.002833`

### tick `17986`, seconds `94.00`, LSTM delta `-0.2234`

Top all feature movements:
- `lag_10__CT_place_TSIDELOWER`: contribution `-0.030724`
- `lag_00__kill_diff_last_3s`: contribution `-0.010201`
- `lag_02__T4__is_scoped`: contribution `-0.009157`
- `lag_06__CT4__flash_duration`: contribution `-0.008378`
- `lag_00__T_kills_last_3s`: contribution `-0.007250`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `-0.008378`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002350`

### tick `18114`, seconds `96.00`, LSTM delta `+0.2123`

Top all feature movements:
- `lag_14__CT_place_TSIDELOWER`: contribution `+0.032821`
- `lag_00__CT_defusing_count`: contribution `+0.022054`
- `lag_10__CT4__flash_duration`: contribution `+0.018833`
- `lag_00__kill_diff_last_3s`: contribution `-0.010201`
- `lag_00__CT_kills_last_3s`: contribution `-0.008656`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.018833`
- `lag_02__T_flash_alpha_mean`: contribution `+0.007991`
- `lag_13__T4__flash_duration`: contribution `+0.007333`
- `lag_15__CT1__flash_duration`: contribution `+0.005007`
- `lag_10__CT_flash_duration_sum`: contribution `+0.003874`
