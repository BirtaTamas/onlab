# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `10`

## Largest probability jumps

- tick `84742`, seconds `104.00`, LSTM `0.6914`, delta `+0.4768`
- tick `84678`, seconds `103.00`, LSTM `0.1356`, delta `-0.2779`
- tick `82182`, seconds `64.00`, LSTM `0.8581`, delta `+0.2510`
- tick `81766`, seconds `57.50`, LSTM `0.7384`, delta `+0.2189`
- tick `84838`, seconds `105.50`, LSTM `0.8490`, delta `+0.2169`
- tick `83494`, seconds `84.50`, LSTM `0.6730`, delta `-0.1712`
- tick `81798`, seconds `58.00`, LSTM `0.5883`, delta `-0.1502`
- tick `84710`, seconds `103.50`, LSTM `0.2146`, delta `+0.0790`
- tick `81318`, seconds `50.50`, LSTM `0.5504`, delta `+0.0657`
- tick `84774`, seconds `104.50`, LSTM `0.6286`, delta `-0.0628`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005212`, |coef| `0.005212`
- `lag_08__CT_place_ENTRANCE`: coefficient `-0.005091`, |coef| `0.005091`
- `lag_00__CT_place_LONGDOG`: coefficient `0.004009`, |coef| `0.004009`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003827`, |coef| `0.003827`
- `lag_00__T_kills_last_3s`: coefficient `-0.003609`, |coef| `0.003609`
- `lag_12__T_bomb_zone_count`: coefficient `-0.003540`, |coef| `0.003540`
- `lag_00__damage_diff_last_5s`: coefficient `0.003523`, |coef| `0.003523`
- `lag_00__CT_kills_last_3s`: coefficient `0.002963`, |coef| `0.002963`
- `lag_00__CT_defusing_count`: coefficient `0.002919`, |coef| `0.002919`
- `lag_02__T_duck_amount_mean`: coefficient `-0.002888`, |coef| `0.002888`
- `lag_02__T3__duck_amount`: coefficient `-0.002639`, |coef| `0.002639`
- `lag_00__CT4__flash`: coefficient `0.002415`, |coef| `0.002415`
- `lag_10__CT_place_ENTRANCE`: coefficient `0.002347`, |coef| `0.002347`
- `lag_15__T_bomb_zone_count`: coefficient `-0.002229`, |coef| `0.002229`
- `lag_01__T_duck_amount_mean`: coefficient `-0.002223`, |coef| `0.002223`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003827` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.002415` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001944` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001823` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001716` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `0.001398` (raises CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001380` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.001316` (raises CT win probability)
- `lag_02__CT1__flash`: coefficient `-0.001138` (lowers CT win probability)
- `lag_01__CT4__utility_total`: coefficient `0.001125` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005212` (raises CT win probability)
- `lag_08__CT_place_ENTRANCE`: coefficient `-0.005091` (lowers CT win probability)
- `lag_00__CT_place_LONGDOG`: coefficient `0.004009` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003609` (lowers CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `-0.003540` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003523` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002963` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002919` (raises CT win probability)
- `lag_02__T_duck_amount_mean`: coefficient `-0.002888` (lowers CT win probability)
- `lag_02__T3__duck_amount`: coefficient `-0.002639` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `84742`, seconds `104.00`, LSTM delta `+0.4768`

Top all feature movements:
- `lag_08__CT_place_ENTRANCE`: contribution `+0.045167`
- `lag_00__T_flash_alpha_mean`: contribution `+0.023219`
- `lag_10__CT_place_ENTRANCE`: contribution `+0.020821`
- `lag_12__T_bomb_zone_count`: contribution `+0.020609`
- `lag_02__T_duck_amount_mean`: contribution `+0.016797`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.023219`

### tick `84678`, seconds `103.00`, LSTM delta `-0.2779`

Top all feature movements:
- `lag_08__CT_place_ENTRANCE`: contribution `-0.045167`
- `lag_00__kill_diff_last_3s`: contribution `-0.012546`
- `lag_06__CT_place_ENTRANCE`: contribution `-0.012205`
- `lag_00__T_kills_last_3s`: contribution `-0.011435`
- `lag_02__T3__duck_amount`: contribution `-0.009949`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82182`, seconds `64.00`, LSTM delta `+0.2510`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012546`
- `lag_00__T_place_IVY`: contribution `+0.011527`
- `lag_00__CT_kills_last_3s`: contribution `+0.008554`
- `lag_00__damage_diff_last_5s`: contribution `+0.007948`
- `lag_12__CT_shots_fired_sum`: contribution `+0.005195`

Top utility-only movements:
- `lag_00__T4__utility_total`: contribution `+0.003156`

### tick `81766`, seconds `57.50`, LSTM delta `+0.2189`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012546`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.009449`
- `lag_00__CT_kills_last_3s`: contribution `+0.008554`
- `lag_00__damage_diff_last_5s`: contribution `+0.008027`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006978`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84838`, seconds `105.50`, LSTM delta `+0.2169`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.028296`
- `lag_15__T_bomb_zone_count`: contribution `+0.012975`
- `lag_11__CT_place_ENTRANCE`: contribution `+0.012653`
- `lag_13__CT_place_ENTRANCE`: contribution `+0.011191`
- `lag_03__T_flash_alpha_mean`: contribution `+0.011062`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.011062`
