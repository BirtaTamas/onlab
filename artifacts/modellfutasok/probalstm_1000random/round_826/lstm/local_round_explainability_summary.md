# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `70561`, seconds `98.00`, LSTM `0.2037`, delta `-0.3598`
- tick `70369`, seconds `95.00`, LSTM `0.5203`, delta `+0.2906`
- tick `69953`, seconds `88.50`, LSTM `0.2569`, delta `-0.2193`
- tick `69377`, seconds `79.50`, LSTM `0.8378`, delta `+0.1969`
- tick `69409`, seconds `80.00`, LSTM `0.6545`, delta `-0.1833`
- tick `71649`, seconds `115.00`, LSTM `0.2121`, delta `+0.1087`
- tick `64833`, seconds `8.50`, LSTM `0.6333`, delta `+0.0965`
- tick `67745`, seconds `54.00`, LSTM `0.5403`, delta `-0.0837`
- tick `70593`, seconds `98.50`, LSTM `0.1457`, delta `-0.0579`
- tick `69697`, seconds `84.50`, LSTM `0.4923`, delta `-0.0572`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004579`, |coef| `0.004579`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003260`, |coef| `0.003260`
- `lag_08__T3__flash_duration`: coefficient `0.003216`, |coef| `0.003216`
- `lag_00__CT_kills_last_3s`: coefficient `0.003008`, |coef| `0.003008`
- `lag_00__damage_diff_last_5s`: coefficient `0.002811`, |coef| `0.002811`
- `lag_00__T_kills_last_3s`: coefficient `-0.002726`, |coef| `0.002726`
- `lag_11__T_place_SNIPERSNEST`: coefficient `0.002706`, |coef| `0.002706`
- `lag_00__CT_defusing_count`: coefficient `0.002680`, |coef| `0.002680`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002408`, |coef| `0.002408`
- `lag_13__T_kills_last_3s`: coefficient `0.002133`, |coef| `0.002133`
- `lag_02__T_place_UNDERPASS`: coefficient `0.002108`, |coef| `0.002108`
- `lag_12__T2__duck_amount`: coefficient `-0.002074`, |coef| `0.002074`
- `lag_13__T4__duck_amount`: coefficient `0.002007`, |coef| `0.002007`
- `lag_13__T_place_UNDERPASS`: coefficient `-0.001999`, |coef| `0.001999`
- `lag_09__T_place_SNIPERSNEST`: coefficient `-0.001920`, |coef| `0.001920`

## Top 10 utility ridge features

- `lag_08__T3__flash_duration`: coefficient `0.003216` (raises CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.001721` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001519` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001503` (lowers CT win probability)
- `lag_10__T4__molly`: coefficient `0.001349` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001258` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001234` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.001202` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001119` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `0.001115` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004579` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003260` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003008` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002811` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002726` (lowers CT win probability)
- `lag_11__T_place_SNIPERSNEST`: coefficient `0.002706` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002680` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002408` (raises CT win probability)
- `lag_13__T_kills_last_3s`: coefficient `0.002133` (raises CT win probability)
- `lag_02__T_place_UNDERPASS`: coefficient `0.002108` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `70561`, seconds `98.00`, LSTM delta `-0.3598`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.022042`
- `lag_08__T3__flash_duration`: contribution `-0.018883`
- `lag_00__CT_duck_amount_mean`: contribution `-0.009762`
- `lag_00__CT_kills_last_3s`: contribution `-0.008684`
- `lag_00__T_kills_last_3s`: contribution `-0.008638`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.018883`
- `lag_09__T_A_site_active_infernos`: contribution `-0.004523`

### tick `70369`, seconds `95.00`, LSTM delta `+0.2906`

Top all feature movements:
- `lag_11__T_place_SNIPERSNEST`: contribution `+0.048093`
- `lag_10__T_place_SNIPERSNEST`: contribution `+0.033760`
- `lag_13__T_place_JUNGLE`: contribution `+0.020697`
- `lag_14__T_place_JUNGLE`: contribution `+0.018830`
- `lag_00__kill_diff_last_3s`: contribution `+0.011021`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.008823`
- `lag_13__T3__flash_duration`: contribution `+0.006006`

### tick `69953`, seconds `88.50`, LSTM delta `-0.2193`

Top all feature movements:
- `lag_09__T_place_SNIPERSNEST`: contribution `-0.034118`
- `lag_00__T_place_JUNGLE`: contribution `-0.015499`
- `lag_03__T_place_JUNGLE`: contribution `-0.015132`
- `lag_11__T_place_JUNGLE`: contribution `-0.013125`
- `lag_00__kill_diff_last_3s`: contribution `-0.011021`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.007245`
- `lag_00__T_flash_duration_sum`: contribution `-0.003040`

### tick `69377`, seconds `79.50`, LSTM delta `+0.1969`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011021`
- `lag_00__CT_kills_last_3s`: contribution `+0.008684`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008365`
- `lag_01__T_place_CONNECTOR`: contribution `+0.006687`
- `lag_14__T_place_CONNECTOR`: contribution `+0.006045`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `69409`, seconds `80.00`, LSTM delta `-0.1833`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.011712`
- `lag_00__kill_diff_last_3s`: contribution `-0.011021`
- `lag_00__T_kills_last_3s`: contribution `-0.008638`
- `lag_13__T4__duck_amount`: contribution `-0.007423`
- `lag_01__T_place_CONNECTOR`: contribution `-0.006687`

Top utility-only movements:
- No utility movement among the top local contributors.
