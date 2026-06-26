# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `65663`, seconds `78.50`, LSTM `0.1402`, delta `-0.2582`
- tick `65599`, seconds `77.50`, LSTM `0.3534`, delta `+0.2373`
- tick `64927`, seconds `67.00`, LSTM `0.1005`, delta `-0.1117`
- tick `65855`, seconds `81.50`, LSTM `0.0892`, delta `-0.1012`
- tick `65567`, seconds `77.00`, LSTM `0.1161`, delta `+0.0697`
- tick `65887`, seconds `82.00`, LSTM `0.0251`, delta `-0.0641`
- tick `65791`, seconds `80.50`, LSTM `0.2273`, delta `+0.0625`
- tick `65631`, seconds `78.00`, LSTM `0.3984`, delta `+0.0450`
- tick `64767`, seconds `64.50`, LSTM `0.2070`, delta `+0.0399`
- tick `65823`, seconds `81.00`, LSTM `0.1904`, delta `-0.0369`

## Top 15 local ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.002374`, |coef| `0.002374`
- `lag_00__kill_diff_last_3s`: coefficient `0.001911`, |coef| `0.001911`
- `lag_00__CT5__flash_duration`: coefficient `-0.001759`, |coef| `0.001759`
- `lag_05__T1__is_scoped`: coefficient `0.001666`, |coef| `0.001666`
- `lag_15__CT5__flash_duration`: coefficient `-0.001532`, |coef| `0.001532`
- `lag_00__damage_diff_last_5s`: coefficient `0.001508`, |coef| `0.001508`
- `lag_10__T_place_CONNECTOR`: coefficient `0.001500`, |coef| `0.001500`
- `lag_13__CT5__flash_duration`: coefficient `0.001488`, |coef| `0.001488`
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001462`, |coef| `0.001462`
- `lag_00__T_kills_last_3s`: coefficient `-0.001412`, |coef| `0.001412`
- `lag_02__CT5__flash_duration`: coefficient `0.001399`, |coef| `0.001399`
- `lag_00__CT_place_UNDERPASS`: coefficient `-0.001394`, |coef| `0.001394`
- `lag_03__T_place_CONNECTOR`: coefficient `0.001380`, |coef| `0.001380`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.001361`, |coef| `0.001361`
- `lag_14__T_B_site_active_infernos`: coefficient `0.001361`, |coef| `0.001361`

## Top 10 utility ridge features

- `lag_00__CT5__flash_duration`: coefficient `-0.001759` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.001532` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `0.001488` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.001399` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.001361` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000886` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000809` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `0.000790` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000785` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000765` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_STAIRS`: coefficient `0.002374` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001911` (raises CT win probability)
- `lag_05__T1__is_scoped`: coefficient `0.001666` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001508` (raises CT win probability)
- `lag_10__T_place_CONNECTOR`: coefficient `0.001500` (raises CT win probability)
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001462` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001412` (lowers CT win probability)
- `lag_00__CT_place_UNDERPASS`: coefficient `-0.001394` (lowers CT win probability)
- `lag_03__T_place_CONNECTOR`: coefficient `0.001380` (raises CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `-0.001361` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65663`, seconds `78.50`, LSTM delta `-0.2582`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `-0.018476`
- `lag_15__CT5__flash_duration`: contribution `-0.010800`
- `lag_02__CT5__flash_duration`: contribution `-0.009857`
- `lag_05__T1__is_scoped`: contribution `-0.009520`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.008082`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.010800`
- `lag_02__CT5__flash_duration`: contribution `-0.009857`

### tick `65599`, seconds `77.50`, LSTM delta `+0.2373`

Top all feature movements:
- `lag_00__CT5__flash_duration`: contribution `+0.012399`
- `lag_13__CT5__flash_duration`: contribution `+0.010483`
- `lag_05__T1__is_scoped`: contribution `+0.009520`
- `lag_14__T_B_site_active_infernos`: contribution `+0.007695`
- `lag_14__CT_place_UNDERPASS`: contribution `+0.007331`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.012399`
- `lag_13__CT5__flash_duration`: contribution `+0.010483`
- `lag_14__T_B_site_active_infernos`: contribution `+0.007695`
- `lag_14__T_active_infernos`: contribution `+0.003268`

### tick `64927`, seconds `67.00`, LSTM delta `-0.1117`

Top all feature movements:
- `lag_11__CT_place_STAIRS`: contribution `-0.007394`
- `lag_05__CT_place_STAIRS`: contribution `-0.006820`
- `lag_12__CT_place_SHOP`: contribution `-0.005116`
- `lag_05__T_place_UNDERPASS`: contribution `-0.004648`
- `lag_00__kill_diff_last_3s`: contribution `-0.004600`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65855`, seconds `81.50`, LSTM delta `-0.1012`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `-0.009633`
- `lag_14__T_place_CONNECTOR`: contribution `-0.007079`
- `lag_00__T_place_CONNECTOR`: contribution `+0.006543`
- `lag_09__T_place_CONNECTOR`: contribution `-0.004002`
- `lag_15__CT_place_SHOP`: contribution `-0.003778`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `-0.003178`

### tick `65567`, seconds `77.00`, LSTM delta `+0.0697`

Top all feature movements:
- `lag_03__T_place_CONNECTOR`: contribution `+0.006681`
- `lag_00__T_place_CONNECTOR`: contribution `+0.006543`
- `lag_12__CT5__flash_duration`: contribution `+0.004990`
- `lag_00__kill_diff_last_3s`: contribution `+0.004600`
- `lag_09__T_place_CONNECTOR`: contribution `+0.004002`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.004990`
- `lag_13__T_B_site_active_infernos`: contribution `+0.002742`
- `lag_06__T_utility_damage_last_5s`: contribution `+0.001804`
