# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `20`

## Largest probability jumps

- tick `183992`, seconds `66.00`, LSTM `0.0295`, delta `-0.1960`
- tick `183736`, seconds `62.00`, LSTM `0.3882`, delta `-0.1655`
- tick `183768`, seconds `62.50`, LSTM `0.3009`, delta `-0.0874`
- tick `183928`, seconds `65.00`, LSTM `0.2676`, delta `-0.0595`
- tick `181240`, seconds `23.00`, LSTM `0.4505`, delta `+0.0476`
- tick `183640`, seconds `60.50`, LSTM `0.5576`, delta `+0.0433`
- tick `183960`, seconds `65.50`, LSTM `0.2255`, delta `-0.0421`
- tick `183896`, seconds `64.50`, LSTM `0.3271`, delta `+0.0361`
- tick `183832`, seconds `63.50`, LSTM `0.2646`, delta `-0.0350`
- tick `181016`, seconds `19.50`, LSTM `0.4000`, delta `-0.0338`

## Top 15 local ridge features

- `lag_00__CT_place_QUAD`: coefficient `0.002060`, |coef| `0.002060`
- `lag_12__T_place_ARCH`: coefficient `-0.001855`, |coef| `0.001855`
- `lag_00__T_place_ARCH`: coefficient `0.001698`, |coef| `0.001698`
- `lag_11__CT_place_QUAD`: coefficient `-0.001621`, |coef| `0.001621`
- `lag_14__T_place_ARCH`: coefficient `-0.001461`, |coef| `0.001461`
- `lag_08__CT_place_QUAD`: coefficient `0.001460`, |coef| `0.001460`
- `lag_05__T_place_ARCH`: coefficient `0.001239`, |coef| `0.001239`
- `lag_04__bomb_events_last_5s`: coefficient `0.001230`, |coef| `0.001230`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001209`, |coef| `0.001209`
- `lag_06__T_place_ARCH`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_00__T_kills_last_3s`: coefficient `-0.001060`, |coef| `0.001060`
- `lag_03__CT_place_QUAD`: coefficient `-0.001017`, |coef| `0.001017`
- `lag_04__CT2__is_walking`: coefficient `0.001011`, |coef| `0.001011`
- `lag_00__kill_diff_last_3s`: coefficient `0.000999`, |coef| `0.000999`
- `lag_13__T_place_ARCH`: coefficient `-0.000960`, |coef| `0.000960`

## Top 10 utility ridge features

- `lag_08__T_A_site_active_infernos`: coefficient `0.000678` (raises CT win probability)
- `lag_14__CT1__smoke`: coefficient `0.000534` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.000514` (raises CT win probability)
- `lag_08__T_active_infernos`: coefficient `0.000478` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000431` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.000426` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000423` (lowers CT win probability)
- `lag_02__T5__smoke`: coefficient `0.000401` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000385` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000380` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_QUAD`: coefficient `0.002060` (raises CT win probability)
- `lag_12__T_place_ARCH`: coefficient `-0.001855` (lowers CT win probability)
- `lag_00__T_place_ARCH`: coefficient `0.001698` (raises CT win probability)
- `lag_11__CT_place_QUAD`: coefficient `-0.001621` (lowers CT win probability)
- `lag_14__T_place_ARCH`: coefficient `-0.001461` (lowers CT win probability)
- `lag_08__CT_place_QUAD`: coefficient `0.001460` (raises CT win probability)
- `lag_05__T_place_ARCH`: coefficient `0.001239` (raises CT win probability)
- `lag_04__bomb_events_last_5s`: coefficient `0.001230` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001209` (raises CT win probability)
- `lag_06__T_place_ARCH`: coefficient `-0.001099` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `183992`, seconds `66.00`, LSTM delta `-0.1960`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `-0.015802`
- `lag_14__T_place_ARCH`: contribution `-0.013595`
- `lag_11__CT_place_QUAD`: contribution `-0.012779`
- `lag_05__T_place_ARCH`: contribution `-0.011530`
- `lag_08__CT_place_QUAD`: contribution `-0.011507`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `183736`, seconds `62.00`, LSTM delta `-0.1655`

Top all feature movements:
- `lag_12__T_place_ARCH`: contribution `-0.017254`
- `lag_00__CT_place_QUAD`: contribution `-0.016238`
- `lag_06__T_place_ARCH`: contribution `-0.010225`
- `lag_03__CT_place_QUAD`: contribution `-0.008013`
- `lag_01__CT_shots_fired_sum`: contribution `-0.006722`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `-0.002018`

### tick `183768`, seconds `62.50`, LSTM delta `-0.0874`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `-0.015802`
- `lag_13__T_place_ARCH`: contribution `-0.008934`
- `lag_01__CT_place_QUAD`: contribution `-0.007099`
- `lag_04__CT_place_QUAD`: contribution `-0.005917`
- `lag_03__CT_shots_fired_sum`: contribution `-0.003006`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `183928`, seconds `65.00`, LSTM delta `-0.0595`

Top all feature movements:
- `lag_12__T_place_ARCH`: contribution `-0.017254`
- `lag_05__T_place_ARCH`: contribution `-0.011530`
- `lag_00__kill_diff_last_3s`: contribution `+0.002406`
- `lag_12__CT5__duck_amount`: contribution `-0.002405`
- `lag_07__CT_shots_fired_sum`: contribution `-0.002044`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `181240`, seconds `23.00`, LSTM delta `+0.0476`

Top all feature movements:
- `lag_10__CT_place_QUAD`: contribution `+0.006243`
- `lag_15__CT4__flash_duration`: contribution `+0.002306`
- `lag_01__CT3__flash_duration`: contribution `+0.001967`
- `lag_00__CT2__is_walking`: contribution `+0.001948`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.001844`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `+0.002306`
- `lag_01__CT3__flash_duration`: contribution `+0.001967`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.001844`
- `lag_05__T3__flash_duration`: contribution `+0.001791`
- `lag_12__CT1__flash_duration`: contribution `+0.001711`
