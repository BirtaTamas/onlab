# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `42338`, seconds `120.50`, LSTM `0.6630`, delta `+0.4203`
- tick `37570`, seconds `46.00`, LSTM `0.6376`, delta `-0.2492`
- tick `37794`, seconds `49.50`, LSTM `0.7747`, delta `+0.2162`
- tick `42402`, seconds `121.50`, LSTM `0.8235`, delta `+0.2051`
- tick `38050`, seconds `53.50`, LSTM `0.7042`, delta `-0.1753`
- tick `39874`, seconds `82.00`, LSTM `0.4141`, delta `-0.1508`
- tick `40482`, seconds `91.50`, LSTM `0.3457`, delta `+0.1135`
- tick `36770`, seconds `33.50`, LSTM `0.9137`, delta `+0.1012`
- tick `37634`, seconds `47.00`, LSTM `0.5770`, delta `-0.0876`
- tick `35746`, seconds `17.50`, LSTM `0.9013`, delta `+0.0700`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.008184`, |coef| `0.008184`
- `lag_00__CT_defusing_count`: coefficient `0.006241`, |coef| `0.006241`
- `lag_00__T_place_JUNGLE`: coefficient `-0.005655`, |coef| `0.005655`
- `lag_00__kill_diff_last_3s`: coefficient `0.005135`, |coef| `0.005135`
- `lag_00__CT_kills_last_3s`: coefficient `0.004355`, |coef| `0.004355`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004186`, |coef| `0.004186`
- `lag_02__T_flash_alpha_mean`: coefficient `-0.003910`, |coef| `0.003910`
- `lag_01__CT_defusing_count`: coefficient `0.003534`, |coef| `0.003534`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003348`, |coef| `0.003348`
- `lag_00__T2__alive`: coefficient `-0.003234`, |coef| `0.003234`
- `lag_01__T_place_CTSPAWN`: coefficient `-0.003166`, |coef| `0.003166`
- `lag_00__CT_velocity_mean`: coefficient `-0.002941`, |coef| `0.002941`
- `lag_00__damage_diff_last_5s`: coefficient `0.002768`, |coef| `0.002768`
- `lag_04__CT1__smoke`: coefficient `-0.002750`, |coef| `0.002750`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.002534`, |coef| `0.002534`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.008184` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004186` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.003910` (lowers CT win probability)
- `lag_04__CT1__smoke`: coefficient `-0.002750` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.002344` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002216` (lowers CT win probability)
- `lag_03__CT1__smoke`: coefficient `-0.001614` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001418` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001407` (raises CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001264` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.006241` (raises CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.005655` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005135` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004355` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.003534` (raises CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003348` (lowers CT win probability)
- `lag_00__T2__alive`: coefficient `-0.003234` (lowers CT win probability)
- `lag_01__T_place_CTSPAWN`: coefficient `-0.003166` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002941` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002768` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `42338`, seconds `120.50`, LSTM delta `+0.4203`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `+0.073250`
- `lag_00__T_flash_alpha_mean`: contribution `+0.049656`
- `lag_00__CT_kills_last_3s`: contribution `+0.012573`
- `lag_00__kill_diff_last_3s`: contribution `+0.012360`
- `lag_13__CT_duck_amount_mean`: contribution `+0.010416`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.049656`
- `lag_04__CT1__smoke`: contribution `+0.005961`
- `lag_00__T2__molly`: contribution `+0.005223`

### tick `37570`, seconds `46.00`, LSTM delta `-0.2492`

Top all feature movements:
- `lag_09__T_place_STAIRS`: contribution `-0.045240`
- `lag_03__T_place_STAIRS`: contribution `-0.037995`
- `lag_02__T_place_STAIRS`: contribution `-0.030954`
- `lag_00__kill_diff_last_3s`: contribution `-0.012360`
- `lag_11__T_velocity_mean`: contribution `-0.008490`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37794`, seconds `49.50`, LSTM delta `+0.2162`

Top all feature movements:
- `lag_09__T_place_STAIRS`: contribution `+0.045240`
- `lag_12__T_place_STAIRS`: contribution `+0.030153`
- `lag_10__T_place_STAIRS`: contribution `+0.014327`
- `lag_00__CT_kills_last_3s`: contribution `+0.012573`
- `lag_00__kill_diff_last_3s`: contribution `+0.012360`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.006331`

### tick `42402`, seconds `121.50`, LSTM delta `+0.2051`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.060497`
- `lag_02__T_flash_alpha_mean`: contribution `+0.023724`
- `lag_02__T_place_JUNGLE`: contribution `+0.014307`
- `lag_12__CT_duck_amount_mean`: contribution `+0.007840`
- `lag_13__CT_duck_amount_mean`: contribution `+0.007478`

Top utility-only movements:
- `lag_02__T_flash_alpha_mean`: contribution `+0.023724`
- `lag_02__T2__molly`: contribution `+0.002332`
- `lag_06__CT1__smoke`: contribution `+0.002303`

### tick `38050`, seconds `53.50`, LSTM delta `-0.1753`

Top all feature movements:
- `lag_14__CT_place_LADDER`: contribution `-0.023339`
- `lag_00__kill_diff_last_3s`: contribution `-0.012360`
- `lag_01__T2__flash_duration`: contribution `-0.006331`
- `lag_00__T_kills_last_3s`: contribution `-0.006275`
- `lag_02__CT_kills_last_3s`: contribution `-0.006084`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `-0.006331`
- `lag_00__CT4__flash`: contribution `-0.003103`
- `lag_09__T2__flash_duration`: contribution `-0.002247`
