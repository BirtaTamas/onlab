# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `4764`, seconds `55.00`, LSTM `0.9090`, delta `+0.1405`
- tick `4924`, seconds `57.50`, LSTM `0.8159`, delta `-0.1123`
- tick `4700`, seconds `54.00`, LSTM `0.7899`, delta `-0.1048`
- tick `4188`, seconds `46.00`, LSTM `0.7651`, delta `+0.0995`
- tick `4476`, seconds `50.50`, LSTM `0.9161`, delta `+0.0917`
- tick `6204`, seconds `77.50`, LSTM `0.8140`, delta `+0.0681`
- tick `2108`, seconds `13.50`, LSTM `0.5971`, delta `+0.0612`
- tick `5084`, seconds `60.00`, LSTM `0.7633`, delta `-0.0540`
- tick `2172`, seconds `14.50`, LSTM `0.6553`, delta `+0.0474`
- tick `2204`, seconds `15.00`, LSTM `0.7027`, delta `+0.0474`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003744`, |coef| `0.003744`
- `lag_00__CT_kills_last_3s`: coefficient `0.003195`, |coef| `0.003195`
- `lag_00__damage_diff_last_5s`: coefficient `0.002835`, |coef| `0.002835`
- `lag_14__CT_place_TRAMP`: coefficient `0.002157`, |coef| `0.002157`
- `lag_03__CT_place_LIBRARY`: coefficient `0.002007`, |coef| `0.002007`
- `lag_00__CT_damage_last_5s`: coefficient `0.001939`, |coef| `0.001939`
- `lag_13__T2__is_walking`: coefficient `0.001714`, |coef| `0.001714`
- `lag_09__CT_kills_last_3s`: coefficient `0.001644`, |coef| `0.001644`
- `lag_00__CT_place_BANANA`: coefficient `0.001634`, |coef| `0.001634`
- `lag_00__T4__alive`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_01__CT_kills_last_3s`: coefficient `0.001577`, |coef| `0.001577`
- `lag_07__CT_place_BANANA`: coefficient `0.001437`, |coef| `0.001437`
- `lag_00__T_kills_last_3s`: coefficient `-0.001423`, |coef| `0.001423`
- `lag_00__T4__smoke`: coefficient `-0.001410`, |coef| `0.001410`
- `lag_03__T_place_TRAMP`: coefficient `0.001403`, |coef| `0.001403`

## Top 10 utility ridge features

- `lag_00__T4__smoke`: coefficient `-0.001410` (lowers CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.000903` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000896` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.000755` (lowers CT win probability)
- `lag_09__T4__smoke`: coefficient `-0.000711` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.000634` (raises CT win probability)
- `lag_02__T4__smoke`: coefficient `-0.000615` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000608` (lowers CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.000584` (raises CT win probability)
- `lag_07__T2__smoke`: coefficient `0.000580` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003744` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003195` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002835` (raises CT win probability)
- `lag_14__CT_place_TRAMP`: coefficient `0.002157` (raises CT win probability)
- `lag_03__CT_place_LIBRARY`: coefficient `0.002007` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001939` (raises CT win probability)
- `lag_13__T2__is_walking`: coefficient `0.001714` (raises CT win probability)
- `lag_09__CT_kills_last_3s`: coefficient `0.001644` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001634` (raises CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001594` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4764`, seconds `55.00`, LSTM delta `+0.1405`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009225`
- `lag_00__kill_diff_last_3s`: contribution `+0.009013`
- `lag_00__damage_diff_last_5s`: contribution `+0.006396`
- `lag_09__CT_kills_last_3s`: contribution `+0.004748`
- `lag_00__CT_damage_last_5s`: contribution `+0.004227`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4924`, seconds `57.50`, LSTM delta `-0.1123`

Top all feature movements:
- `lag_03__CT_place_LIBRARY`: contribution `-0.012868`
- `lag_00__kill_diff_last_3s`: contribution `-0.009013`
- `lag_00__damage_diff_last_5s`: contribution `-0.006396`
- `lag_02__T_duck_amount_mean`: contribution `-0.005734`
- `lag_00__CT_place_BANANA`: contribution `-0.004838`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4700`, seconds `54.00`, LSTM delta `-0.1048`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009013`
- `lag_15__T5__duck_amount`: contribution `-0.005259`
- `lag_00__CT_place_BANANA`: contribution `-0.004838`
- `lag_01__CT_kills_last_3s`: contribution `-0.004554`
- `lag_00__T_kills_last_3s`: contribution `-0.004507`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4188`, seconds `46.00`, LSTM delta `+0.0995`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009225`
- `lag_00__kill_diff_last_3s`: contribution `+0.009013`
- `lag_13__T2__is_walking`: contribution `+0.003936`
- `lag_00__T4__alive`: contribution `+0.003917`
- `lag_00__damage_diff_last_5s`: contribution `+0.003901`

Top utility-only movements:
- `lag_00__T4__smoke`: contribution `+0.003067`

### tick `4476`, seconds `50.50`, LSTM delta `+0.0917`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009225`
- `lag_00__kill_diff_last_3s`: contribution `+0.009013`
- `lag_00__damage_diff_last_5s`: contribution `+0.006396`
- `lag_09__CT_kills_last_3s`: contribution `+0.004748`
- `lag_15__T5__duck_amount`: contribution `+0.004596`

Top utility-only movements:
- No utility movement among the top local contributors.
