# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `124452`, seconds `56.50`, LSTM `0.3989`, delta `-0.3245`
- tick `123908`, seconds `48.00`, LSTM `0.1511`, delta `-0.3222`
- tick `124260`, seconds `53.50`, LSTM `0.7050`, delta `+0.2559`
- tick `124580`, seconds `58.50`, LSTM `0.7223`, delta `+0.2184`
- tick `124036`, seconds `50.00`, LSTM `0.1634`, delta `+0.1483`
- tick `124548`, seconds `58.00`, LSTM `0.5039`, delta `+0.1215`
- tick `124196`, seconds `52.50`, LSTM `0.4559`, delta `+0.0949`
- tick `123940`, seconds `48.50`, LSTM `0.0606`, delta `-0.0905`
- tick `124612`, seconds `59.00`, LSTM `0.7905`, delta `+0.0682`
- tick `124100`, seconds `51.00`, LSTM `0.2886`, delta `+0.0642`

## Top 15 local ridge features

- `lag_00__T_place_JUNGLE`: coefficient `-0.003239`, |coef| `0.003239`
- `lag_09__T_place_CONNECTOR`: coefficient `-0.003087`, |coef| `0.003087`
- `lag_00__kill_diff_last_3s`: coefficient `0.002978`, |coef| `0.002978`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002771`, |coef| `0.002771`
- `lag_08__T_place_CONNECTOR`: coefficient `-0.002562`, |coef| `0.002562`
- `lag_03__CT_place_BACKALLEY`: coefficient `0.002523`, |coef| `0.002523`
- `lag_11__CT_place_BACKALLEY`: coefficient `-0.002468`, |coef| `0.002468`
- `lag_00__T_kills_last_3s`: coefficient `-0.002347`, |coef| `0.002347`
- `lag_12__bomb_events_last_5s`: coefficient `-0.002330`, |coef| `0.002330`
- `lag_13__bomb_events_last_5s`: coefficient `-0.002253`, |coef| `0.002253`
- `lag_07__CT_place_JUNGLE`: coefficient `0.002212`, |coef| `0.002212`
- `lag_15__T_place_JUNGLE`: coefficient `-0.002160`, |coef| `0.002160`
- `lag_10__CT_place_UNDERPASS`: coefficient `0.002139`, |coef| `0.002139`
- `lag_10__T_place_CONNECTOR`: coefficient `-0.002038`, |coef| `0.002038`
- `lag_06__T_place_CONNECTOR`: coefficient `-0.002030`, |coef| `0.002030`

## Top 10 utility ridge features

- `lag_00__T4__molly`: coefficient `0.001022` (raises CT win probability)
- `lag_01__T4__molly`: coefficient `0.000593` (raises CT win probability)
- `lag_07__T5__smoke`: coefficient `-0.000578` (lowers CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.000542` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000532` (raises CT win probability)
- `lag_09__T1__molly`: coefficient `-0.000524` (lowers CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `0.000482` (raises CT win probability)
- `lag_07__T5__molly`: coefficient `-0.000481` (lowers CT win probability)
- `lag_06__T1__smoke`: coefficient `-0.000468` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.000468` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_JUNGLE`: coefficient `-0.003239` (lowers CT win probability)
- `lag_09__T_place_CONNECTOR`: coefficient `-0.003087` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002978` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002771` (raises CT win probability)
- `lag_08__T_place_CONNECTOR`: coefficient `-0.002562` (lowers CT win probability)
- `lag_03__CT_place_BACKALLEY`: coefficient `0.002523` (raises CT win probability)
- `lag_11__CT_place_BACKALLEY`: coefficient `-0.002468` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002347` (lowers CT win probability)
- `lag_12__bomb_events_last_5s`: coefficient `-0.002330` (lowers CT win probability)
- `lag_13__bomb_events_last_5s`: coefficient `-0.002253` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `124452`, seconds `56.50`, LSTM delta `-0.3245`

Top all feature movements:
- `lag_03__CT_place_BACKALLEY`: contribution `-0.037828`
- `lag_11__CT_place_BACKALLEY`: contribution `-0.037006`
- `lag_15__T_place_JUNGLE`: contribution `-0.027980`
- `lag_06__T_place_JUNGLE`: contribution `-0.024932`
- `lag_00__CT_place_JUNGLE`: contribution `-0.017774`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123908`, seconds `48.00`, LSTM delta `-0.3222`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.017774`
- `lag_09__T_place_CONNECTOR`: contribution `-0.014947`
- `lag_07__CT_place_JUNGLE`: contribution `-0.014191`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.012402`
- `lag_14__CT_place_JUNGLE`: contribution `-0.011359`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124260`, seconds `53.50`, LSTM delta `+0.2559`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `+0.041964`
- `lag_05__CT_place_BACKALLEY`: contribution `+0.027911`
- `lag_08__T_place_CONNECTOR`: contribution `+0.012409`
- `lag_09__T_place_JUNGLE`: contribution `+0.012115`
- `lag_10__T_place_CONNECTOR`: contribution `+0.009868`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124580`, seconds `58.50`, LSTM delta `+0.2184`

Top all feature movements:
- `lag_01__T_place_STAIRS`: contribution `+0.037594`
- `lag_00__CT_place_SIDEALLEY`: contribution `+0.034153`
- `lag_15__CT_place_BACKALLEY`: contribution `+0.026149`
- `lag_09__T_place_CONNECTOR`: contribution `+0.014947`
- `lag_07__CT_place_BACKALLEY`: contribution `+0.011238`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124036`, seconds `50.00`, LSTM delta `+0.1483`

Top all feature movements:
- `lag_02__T_place_JUNGLE`: contribution `+0.010516`
- `lag_10__T_place_CONNECTOR`: contribution `-0.009868`
- `lag_03__T_place_CONNECTOR`: contribution `+0.009470`
- `lag_06__CT_place_JUNGLE`: contribution `+0.008000`
- `lag_00__kill_diff_last_3s`: contribution `+0.007167`

Top utility-only movements:
- No utility movement among the top local contributors.
