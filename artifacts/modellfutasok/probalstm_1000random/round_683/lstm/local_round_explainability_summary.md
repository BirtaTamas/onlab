# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `2`

## Largest probability jumps

- tick `16315`, seconds `72.00`, LSTM `0.5558`, delta `-0.2955`
- tick `16379`, seconds `73.00`, LSTM `0.2463`, delta `-0.2569`
- tick `16027`, seconds `67.50`, LSTM `0.8329`, delta `+0.1213`
- tick `15995`, seconds `67.00`, LSTM `0.7116`, delta `+0.0915`
- tick `15483`, seconds `59.00`, LSTM `0.7306`, delta `-0.0904`
- tick `15547`, seconds `60.00`, LSTM `0.6642`, delta `-0.0740`
- tick `16411`, seconds `73.50`, LSTM `0.1854`, delta `-0.0609`
- tick `15259`, seconds `55.50`, LSTM `0.8626`, delta `-0.0602`
- tick `16443`, seconds `74.00`, LSTM `0.1271`, delta `-0.0583`
- tick `16347`, seconds `72.50`, LSTM `0.5032`, delta `-0.0526`

## Top 15 local ridge features

- `lag_00__T_place_HEAVEN`: coefficient `-0.004826`, |coef| `0.004826`
- `lag_02__T_place_HEAVEN`: coefficient `-0.004009`, |coef| `0.004009`
- `lag_01__T_place_HEAVEN`: coefficient `-0.002924`, |coef| `0.002924`
- `lag_00__damage_diff_last_5s`: coefficient `0.002709`, |coef| `0.002709`
- `lag_00__kill_diff_last_3s`: coefficient `0.002197`, |coef| `0.002197`
- `lag_07__CT_place_MAIN`: coefficient `-0.002179`, |coef| `0.002179`
- `lag_03__T_place_HEAVEN`: coefficient `-0.002048`, |coef| `0.002048`
- `lag_09__CT_place_MAIN`: coefficient `-0.001944`, |coef| `0.001944`
- `lag_00__T_kills_last_3s`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_09__T_place_MAIN`: coefficient `0.001784`, |coef| `0.001784`
- `lag_07__CT_place_CANAL`: coefficient `0.001723`, |coef| `0.001723`
- `lag_01__damage_diff_last_5s`: coefficient `0.001609`, |coef| `0.001609`
- `lag_10__T_A_site_active_infernos`: coefficient `0.001566`, |coef| `0.001566`
- `lag_00__CT_place_MIDDLE`: coefficient `0.001560`, |coef| `0.001560`
- `lag_11__T_bomb_zone_count`: coefficient `-0.001559`, |coef| `0.001559`

## Top 10 utility ridge features

- `lag_10__T_A_site_active_infernos`: coefficient `0.001566` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001452` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.001253` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.001138` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001126` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.001105` (raises CT win probability)
- `lag_02__CT3__flash`: coefficient `0.001082` (raises CT win probability)
- `lag_09__T_active_infernos`: coefficient `0.000885` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000870` (raises CT win probability)
- `lag_02__CT3__utility_total`: coefficient `0.000852` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HEAVEN`: coefficient `-0.004826` (lowers CT win probability)
- `lag_02__T_place_HEAVEN`: coefficient `-0.004009` (lowers CT win probability)
- `lag_01__T_place_HEAVEN`: coefficient `-0.002924` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002709` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002197` (raises CT win probability)
- `lag_07__CT_place_MAIN`: coefficient `-0.002179` (lowers CT win probability)
- `lag_03__T_place_HEAVEN`: coefficient `-0.002048` (lowers CT win probability)
- `lag_09__CT_place_MAIN`: coefficient `-0.001944` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001889` (lowers CT win probability)
- `lag_09__T_place_MAIN`: coefficient `0.001784` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `16315`, seconds `72.00`, LSTM delta `-0.2955`

Top all feature movements:
- `lag_00__T_place_HEAVEN`: contribution `-0.059218`
- `lag_07__CT_place_MAIN`: contribution `-0.014672`
- `lag_09__T_place_MAIN`: contribution `-0.011534`
- `lag_07__CT_place_CANAL`: contribution `-0.010471`
- `lag_00__damage_diff_last_5s`: contribution `-0.010450`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.005360`
- `lag_08__T_A_site_active_infernos`: contribution `-0.003387`
- `lag_00__CT3__utility_total`: contribution `-0.003224`

### tick `16379`, seconds `73.00`, LSTM delta `-0.2569`

Top all feature movements:
- `lag_02__T_place_HEAVEN`: contribution `-0.049194`
- `lag_09__CT_place_MAIN`: contribution `-0.013089`
- `lag_09__CT_place_CANAL`: contribution `-0.006881`
- `lag_00__damage_diff_last_5s`: contribution `-0.006111`
- `lag_00__T_kills_last_3s`: contribution `-0.005985`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.004660`
- `lag_02__CT3__flash`: contribution `-0.003996`
- `lag_02__CT3__utility_total`: contribution `-0.002439`

### tick `16027`, seconds `67.50`, LSTM delta `+0.1213`

Top all feature movements:
- `lag_11__T_bomb_zone_count`: contribution `+0.009076`
- `lag_00__T_place_MAIN`: contribution `+0.006062`
- `lag_00__kill_diff_last_3s`: contribution `+0.005289`
- `lag_15__T_place_FOUNTAIN`: contribution `+0.005126`
- `lag_15__T_place_MAIN`: contribution `+0.004950`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `+0.004660`
- `lag_10__T_active_infernos`: contribution `+0.002301`

### tick `15995`, seconds `67.00`, LSTM delta `+0.0915`

Top all feature movements:
- `lag_14__T_place_MAIN`: contribution `+0.008561`
- `lag_10__T_bomb_zone_count`: contribution `+0.007758`
- `lag_14__T_place_FOUNTAIN`: contribution `+0.005656`
- `lag_00__damage_diff_last_5s`: contribution `+0.004339`
- `lag_09__T_A_site_active_infernos`: contribution `+0.003728`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `+0.003728`
- `lag_09__T_active_infernos`: contribution `+0.001844`

### tick `15483`, seconds `59.00`, LSTM delta `-0.0904`

Top all feature movements:
- `lag_14__T_place_MAIN`: contribution `-0.008561`
- `lag_11__CT_place_BRICKS`: contribution `-0.006993`
- `lag_09__CT_place_CANAL`: contribution `+0.006881`
- `lag_13__CT_place_FOUNTAIN`: contribution `-0.006816`
- `lag_15__T_place_MAIN`: contribution `-0.004950`

Top utility-only movements:
- No utility movement among the top local contributors.
