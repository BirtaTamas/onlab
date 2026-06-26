# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `29626`, seconds `86.50`, LSTM `0.5209`, delta `-0.2045`
- tick `30778`, seconds `104.50`, LSTM `0.8778`, delta `+0.1722`
- tick `30010`, seconds `92.50`, LSTM `0.5773`, delta `+0.1030`
- tick `25626`, seconds `24.00`, LSTM `0.6394`, delta `+0.0928`
- tick `30714`, seconds `103.50`, LSTM `0.6663`, delta `+0.0780`
- tick `26810`, seconds `42.50`, LSTM `0.6783`, delta `-0.0717`
- tick `30170`, seconds `95.00`, LSTM `0.7532`, delta `+0.0599`
- tick `25914`, seconds `28.50`, LSTM `0.7602`, delta `+0.0593`
- tick `30330`, seconds `97.50`, LSTM `0.6031`, delta `-0.0525`
- tick `27610`, seconds `55.00`, LSTM `0.6705`, delta `-0.0506`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003438`, |coef| `0.003438`
- `lag_05__T_place_UNDERPASS`: coefficient `0.003410`, |coef| `0.003410`
- `lag_00__T5__is_scoped`: coefficient `0.003315`, |coef| `0.003315`
- `lag_00__damage_diff_last_5s`: coefficient `0.003276`, |coef| `0.003276`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.002744`, |coef| `0.002744`
- `lag_13__CT_place_JUNGLE`: coefficient `-0.002648`, |coef| `0.002648`
- `lag_04__T_place_PALACEALLEY`: coefficient `-0.002645`, |coef| `0.002645`
- `lag_04__CT_place_JUNGLE`: coefficient `0.002643`, |coef| `0.002643`
- `lag_00__T_kills_last_3s`: coefficient `-0.002602`, |coef| `0.002602`
- `lag_12__T_place_PALACEALLEY`: coefficient `-0.002484`, |coef| `0.002484`
- `lag_02__T4__flash_duration`: coefficient `0.002186`, |coef| `0.002186`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002025`, |coef| `0.002025`
- `lag_00__T_scoped_count`: coefficient `0.002003`, |coef| `0.002003`
- `lag_09__T_bomb_zone_count`: coefficient `0.001987`, |coef| `0.001987`
- `lag_15__CT4__is_walking`: coefficient `0.001862`, |coef| `0.001862`

## Top 10 utility ridge features

- `lag_02__T4__flash_duration`: coefficient `0.002186` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001677` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001477` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001402` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001032` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000870` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000755` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.000737` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000730` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000700` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003438` (raises CT win probability)
- `lag_05__T_place_UNDERPASS`: coefficient `0.003410` (raises CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.003315` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003276` (raises CT win probability)
- `lag_00__CT_place_CONNECTOR`: coefficient `0.002744` (raises CT win probability)
- `lag_13__CT_place_JUNGLE`: coefficient `-0.002648` (lowers CT win probability)
- `lag_04__T_place_PALACEALLEY`: coefficient `-0.002645` (lowers CT win probability)
- `lag_04__CT_place_JUNGLE`: coefficient `0.002643` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002602` (lowers CT win probability)
- `lag_12__T_place_PALACEALLEY`: coefficient `-0.002484` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `29626`, seconds `86.50`, LSTM delta `-0.2045`

Top all feature movements:
- `lag_13__CT_place_JUNGLE`: contribution `-0.016990`
- `lag_04__CT_place_JUNGLE`: contribution `-0.016958`
- `lag_00__T5__is_scoped`: contribution `-0.015812`
- `lag_05__T_place_UNDERPASS`: contribution `-0.013359`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.009814`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30778`, seconds `104.50`, LSTM delta `+0.1722`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014067`
- `lag_02__T4__flash_duration`: contribution `+0.014050`
- `lag_09__T_bomb_zone_count`: contribution `+0.011567`
- `lag_03__T_bomb_zone_count`: contribution `+0.009120`
- `lag_00__kill_diff_last_3s`: contribution `+0.008276`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.014050`
- `lag_02__T3__flash_duration`: contribution `+0.007687`
- `lag_02__T_flash_duration_sum`: contribution `+0.007235`

### tick `30010`, seconds `92.50`, LSTM delta `+0.1030`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008276`
- `lag_00__damage_diff_last_5s`: contribution `+0.007391`
- `lag_00__CT_kills_last_3s`: contribution `+0.005060`
- `lag_06__T_place_PALACEALLEY`: contribution `+0.004619`
- `lag_15__CT4__is_walking`: contribution `+0.004439`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25626`, seconds `24.00`, LSTM delta `+0.0928`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008276`
- `lag_14__T_place_SIDEALLEY`: contribution `+0.007920`
- `lag_00__damage_diff_last_5s`: contribution `+0.007391`
- `lag_00__CT_kills_last_3s`: contribution `+0.005060`
- `lag_11__CT5__flash_duration`: contribution `+0.004735`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.004735`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.002138`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.002023`
- `lag_15__CT2__flash_duration`: contribution `+0.001932`

### tick `30714`, seconds `103.50`, LSTM delta `+0.0780`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.010778`
- `lag_07__T_bomb_zone_count`: contribution `+0.008799`
- `lag_14__CT_place_SNIPERSNEST`: contribution `+0.006441`
- `lag_01__T_bomb_zone_count`: contribution `+0.005581`
- `lag_01__CT_place_CATWALK`: contribution `+0.004243`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.010778`
- `lag_00__T_flash_duration_sum`: contribution `+0.002924`
- `lag_00__T3__flash_duration`: contribution `-0.002150`
