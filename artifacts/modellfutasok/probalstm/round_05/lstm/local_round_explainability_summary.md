# Local Round Explainability

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `2`

## Largest probability jumps

- tick `13853`, seconds `55.00`, LSTM `0.2030`, delta `-0.4072`
- tick `17085`, seconds `105.50`, LSTM `0.8536`, delta `+0.3579`
- tick `16125`, seconds `90.50`, LSTM `0.4206`, delta `-0.3396`
- tick `16029`, seconds `89.00`, LSTM `0.7302`, delta `+0.2596`
- tick `15293`, seconds `77.50`, LSTM `0.3538`, delta `+0.2362`
- tick `16893`, seconds `102.50`, LSTM `0.5677`, delta `+0.1878`
- tick `16349`, seconds `94.00`, LSTM `0.3441`, delta `-0.1542`
- tick `17213`, seconds `107.50`, LSTM `0.9228`, delta `+0.1136`
- tick `16797`, seconds `101.00`, LSTM `0.1948`, delta `-0.1120`
- tick `16829`, seconds `101.50`, LSTM `0.3059`, delta `+0.1111`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005088`, |coef| `0.005088`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004733`, |coef| `0.004733`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003982`, |coef| `0.003982`
- `lag_00__T_kills_last_3s`: coefficient `-0.003933`, |coef| `0.003933`
- `lag_00__damage_diff_last_5s`: coefficient `0.003895`, |coef| `0.003895`
- `lag_15__CT_place_WALKWAY`: coefficient `-0.003064`, |coef| `0.003064`
- `lag_00__CT_defusing_count`: coefficient `0.002956`, |coef| `0.002956`
- `lag_15__CT_place_HEAVEN`: coefficient `0.002837`, |coef| `0.002837`
- `lag_14__T4__duck_amount`: coefficient `0.002816`, |coef| `0.002816`
- `lag_13__CT_place_CANAL`: coefficient `-0.002739`, |coef| `0.002739`
- `lag_12__CT_place_HEAVEN`: coefficient `-0.002529`, |coef| `0.002529`
- `lag_00__alive_diff`: coefficient `0.002526`, |coef| `0.002526`
- `lag_00__CT_kills_last_3s`: coefficient `0.002519`, |coef| `0.002519`
- `lag_00__T_macro_B`: coefficient `-0.002508`, |coef| `0.002508`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002508`, |coef| `0.002508`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003982` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001922` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001842` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001445` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001380` (lowers CT win probability)
- `lag_06__T1__smoke`: coefficient `0.001356` (raises CT win probability)
- `lag_02__CT2__flash`: coefficient `0.001061` (raises CT win probability)
- `lag_02__T_B_site_active_smokes`: coefficient `-0.001006` (lowers CT win probability)
- `lag_02__T_active_smokes`: coefficient `-0.000981` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000968` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005088` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004733` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003933` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003895` (raises CT win probability)
- `lag_15__CT_place_WALKWAY`: coefficient `-0.003064` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002956` (raises CT win probability)
- `lag_15__CT_place_HEAVEN`: coefficient `0.002837` (raises CT win probability)
- `lag_14__T4__duck_amount`: coefficient `0.002816` (raises CT win probability)
- `lag_13__CT_place_CANAL`: coefficient `-0.002739` (lowers CT win probability)
- `lag_12__CT_place_HEAVEN`: coefficient `-0.002529` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `13853`, seconds `55.00`, LSTM delta `-0.4072`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.036168`
- `lag_00__T_kills_last_3s`: contribution `-0.024918`
- `lag_00__kill_diff_last_3s`: contribution `-0.024493`
- `lag_05__CT_place_WALKWAY`: contribution `-0.012113`
- `lag_00__T_shots_fired_sum`: contribution `-0.010765`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17085`, seconds `105.50`, LSTM delta `+0.3579`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.036168`
- `lag_00__T_flash_alpha_mean`: contribution `+0.024158`
- `lag_13__CT_place_BRICKS`: contribution `+0.019102`
- `lag_04__CT_place_BRICKS`: contribution `+0.017058`
- `lag_15__CT_place_BACKOFB`: contribution `+0.013378`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.024158`

### tick `16125`, seconds `90.50`, LSTM delta `-0.3396`

Top all feature movements:
- `lag_15__CT_place_HEAVEN`: contribution `-0.015319`
- `lag_15__CT_place_WALKWAY`: contribution `-0.015041`
- `lag_00__T_kills_last_3s`: contribution `-0.012459`
- `lag_00__kill_diff_last_3s`: contribution `-0.012247`
- `lag_14__T4__duck_amount`: contribution `-0.010414`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16029`, seconds `89.00`, LSTM delta `+0.2596`

Top all feature movements:
- `lag_13__CT_place_CANAL`: contribution `+0.016648`
- `lag_12__CT_place_HEAVEN`: contribution `+0.013657`
- `lag_00__kill_diff_last_3s`: contribution `+0.012247`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009864`
- `lag_00__damage_diff_last_5s`: contribution `+0.008436`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15293`, seconds `77.50`, LSTM delta `+0.2362`

Top all feature movements:
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `+0.028520`
- `lag_00__CT_shots_fired_sum`: contribution `+0.023016`
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.015220`
- `lag_12__CT_place_HEAVEN`: contribution `+0.013657`
- `lag_00__kill_diff_last_3s`: contribution `+0.012247`

Top utility-only movements:
- No utility movement among the top local contributors.
