# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `23227`, seconds `81.50`, LSTM `0.6977`, delta `+0.3388`
- tick `23163`, seconds `80.50`, LSTM `0.3605`, delta `-0.2579`
- tick `23387`, seconds `84.00`, LSTM `0.8994`, delta `+0.2462`
- tick `23483`, seconds `85.50`, LSTM `0.9169`, delta `+0.1867`
- tick `23419`, seconds `84.50`, LSTM `0.7201`, delta `-0.1793`
- tick `22811`, seconds `75.00`, LSTM `0.6632`, delta `+0.1112`
- tick `23259`, seconds `82.00`, LSTM `0.6187`, delta `-0.0790`
- tick `23131`, seconds `80.00`, LSTM `0.6184`, delta `-0.0706`
- tick `22907`, seconds `76.50`, LSTM `0.7556`, delta `+0.0496`
- tick `23323`, seconds `83.00`, LSTM `0.6145`, delta `-0.0419`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002649`, |coef| `0.002649`
- `lag_00__CT_kills_last_3s`: coefficient `0.002170`, |coef| `0.002170`
- `lag_13__T_flashed_players`: coefficient `-0.002166`, |coef| `0.002166`
- `lag_00__damage_diff_last_5s`: coefficient `0.001977`, |coef| `0.001977`
- `lag_05__CT_place_TRUCK`: coefficient `0.001882`, |coef| `0.001882`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001830`, |coef| `0.001830`
- `lag_14__CT_place_TRUCK`: coefficient `-0.001582`, |coef| `0.001582`
- `lag_02__T_shots_fired_sum`: coefficient `0.001520`, |coef| `0.001520`
- `lag_13__CT2__is_scoped`: coefficient `-0.001473`, |coef| `0.001473`
- `lag_07__CT_place_TRUCK`: coefficient `-0.001446`, |coef| `0.001446`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001413`, |coef| `0.001413`
- `lag_00__CT_damage_last_5s`: coefficient `0.001386`, |coef| `0.001386`
- `lag_10__CT2__is_scoped`: coefficient `0.001352`, |coef| `0.001352`
- `lag_13__CT_flashed_players`: coefficient `-0.001336`, |coef| `0.001336`
- `lag_07__T_shots_fired_sum`: coefficient `0.001333`, |coef| `0.001333`

## Top 10 utility ridge features

- `lag_05__CT2__flash_duration`: coefficient `0.001321` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.001216` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.001188` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.001175` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.000823` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000759` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000728` (raises CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000722` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000708` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000681` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002649` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002170` (raises CT win probability)
- `lag_13__T_flashed_players`: coefficient `-0.002166` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001977` (raises CT win probability)
- `lag_05__CT_place_TRUCK`: coefficient `0.001882` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001830` (lowers CT win probability)
- `lag_14__CT_place_TRUCK`: coefficient `-0.001582` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `0.001520` (raises CT win probability)
- `lag_13__CT2__is_scoped`: coefficient `-0.001473` (lowers CT win probability)
- `lag_07__CT_place_TRUCK`: coefficient `-0.001446` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23227`, seconds `81.50`, LSTM delta `+0.3388`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.009605`
- `lag_07__CT_place_TRUCK`: contribution `+0.009330`
- `lag_13__CT2__is_scoped`: contribution `+0.009017`
- `lag_13__T_flashed_players`: contribution `+0.008358`
- `lag_10__CT2__is_scoped`: contribution `+0.008272`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.005096`

### tick `23163`, seconds `80.50`, LSTM delta `-0.2579`

Top all feature movements:
- `lag_13__T_flashed_players`: contribution `-0.012537`
- `lag_05__CT_place_TRUCK`: contribution `-0.012136`
- `lag_14__CT_place_TRUCK`: contribution `-0.010202`
- `lag_00__kill_diff_last_3s`: contribution `-0.006376`
- `lag_13__CT_flashed_players`: contribution `-0.005850`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `-0.005666`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.004148`

### tick `23387`, seconds `84.00`, LSTM delta `+0.2462`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `+0.009852`
- `lag_12__CT_place_TRUCK`: contribution `+0.007887`
- `lag_07__T_shots_fired_sum`: contribution `+0.006995`
- `lag_00__kill_diff_last_3s`: contribution `+0.006376`
- `lag_00__CT_kills_last_3s`: contribution `+0.006265`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `+0.005216`

### tick `23483`, seconds `85.50`, LSTM delta `+0.1867`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `+0.016896`
- `lag_01__T_place_TRUCK`: contribution `+0.011493`
- `lag_01__T_shots_fired_sum`: contribution `+0.010977`
- `lag_01__CT_place_STAIRS`: contribution `+0.008787`
- `lag_00__kill_diff_last_3s`: contribution `+0.006376`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.003254`

### tick `23419`, seconds `84.50`, LSTM delta `-0.1793`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012753`
- `lag_01__CT_place_STAIRS`: contribution `-0.008787`
- `lag_07__T_shots_fired_sum`: contribution `-0.006995`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006872`
- `lag_00__CT_kills_last_3s`: contribution `-0.006265`

Top utility-only movements:
- No utility movement among the top local contributors.
