# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `15`

## Largest probability jumps

- tick `134699`, seconds `43.00`, LSTM `0.7534`, delta `+0.2074`
- tick `135627`, seconds `57.50`, LSTM `0.7940`, delta `-0.1369`
- tick `136203`, seconds `66.50`, LSTM `0.9038`, delta `+0.1369`
- tick `134987`, seconds `47.50`, LSTM `0.8335`, delta `-0.0981`
- tick `134795`, seconds `44.50`, LSTM `0.8872`, delta `+0.0955`
- tick `135531`, seconds `56.00`, LSTM `0.9123`, delta `+0.0884`
- tick `136011`, seconds `63.50`, LSTM `0.6481`, delta `-0.0723`
- tick `134635`, seconds `42.00`, LSTM `0.5534`, delta `-0.0612`
- tick `136139`, seconds `65.50`, LSTM `0.7184`, delta `+0.0559`
- tick `135499`, seconds `55.50`, LSTM `0.8239`, delta `+0.0545`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002601`, |coef| `0.002601`
- `lag_02__CT_flashed_players`: coefficient `0.002501`, |coef| `0.002501`
- `lag_02__CT4__flash_duration`: coefficient `0.002393`, |coef| `0.002393`
- `lag_00__CT_kills_last_3s`: coefficient `0.002241`, |coef| `0.002241`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001750`, |coef| `0.001750`
- `lag_04__CT_place_ARAMP`: coefficient `-0.001602`, |coef| `0.001602`
- `lag_07__T3__flash_duration`: coefficient `0.001576`, |coef| `0.001576`
- `lag_12__CT_place_UNDERA`: coefficient `-0.001573`, |coef| `0.001573`
- `lag_00__CT_place_HOLE`: coefficient `0.001496`, |coef| `0.001496`
- `lag_07__T5__flash_duration`: coefficient `0.001439`, |coef| `0.001439`
- `lag_03__CT_flashed_players`: coefficient `0.001437`, |coef| `0.001437`
- `lag_03__T_place_MIDDOORS`: coefficient `0.001425`, |coef| `0.001425`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001422`, |coef| `0.001422`
- `lag_02__CT_flash_duration_sum`: coefficient `0.001358`, |coef| `0.001358`
- `lag_10__CT_place_MIDDOORS`: coefficient `-0.001349`, |coef| `0.001349`

## Top 10 utility ridge features

- `lag_02__CT4__flash_duration`: coefficient `0.002393` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.001576` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001439` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001358` (raises CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.001258` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.001136` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.000993` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.000871` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000852` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.000848` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002601` (raises CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.002501` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002241` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001750` (lowers CT win probability)
- `lag_04__CT_place_ARAMP`: coefficient `-0.001602` (lowers CT win probability)
- `lag_12__CT_place_UNDERA`: coefficient `-0.001573` (lowers CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001496` (raises CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.001437` (raises CT win probability)
- `lag_03__T_place_MIDDOORS`: coefficient `0.001425` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001422` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `134699`, seconds `43.00`, LSTM delta `+0.2074`

Top all feature movements:
- `lag_02__CT4__flash_duration`: contribution `+0.015065`
- `lag_07__T3__flash_duration`: contribution `+0.011348`
- `lag_02__CT_flashed_players`: contribution `+0.010955`
- `lag_04__CT_place_ARAMP`: contribution `+0.009978`
- `lag_01__CT_place_ARAMP`: contribution `+0.008129`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.015065`
- `lag_07__T3__flash_duration`: contribution `+0.011348`
- `lag_07__T5__flash_duration`: contribution `+0.008067`
- `lag_07__T_flash_duration_sum`: contribution `+0.006639`
- `lag_02__CT_flash_duration_sum`: contribution `+0.004590`

### tick `135627`, seconds `57.50`, LSTM delta `-0.1369`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.006260`
- `lag_03__T_place_MIDDOORS`: contribution `-0.006057`
- `lag_02__CT_flashed_players`: contribution `-0.005477`
- `lag_12__CT_place_UNDERA`: contribution `-0.004804`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.003901`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `-0.003901`
- `lag_02__CT4__flash_duration`: contribution `-0.003680`
- `lag_15__CT4__flash_duration`: contribution `-0.003149`

### tick `136203`, seconds `66.50`, LSTM delta `+0.1369`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.010185`
- `lag_09__CT_shots_fired_sum`: contribution `+0.009154`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008460`
- `lag_09__CT5__shots_fired`: contribution `+0.007095`
- `lag_00__CT_kills_last_3s`: contribution `+0.006469`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `134987`, seconds `47.50`, LSTM delta `-0.0981`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012520`
- `lag_02__CT4__flash_duration`: contribution `-0.008417`
- `lag_00__CT_kills_last_3s`: contribution `-0.006469`
- `lag_04__CT_shots_fired_sum`: contribution `-0.004469`
- `lag_04__CT4__shots_fired`: contribution `-0.004232`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.008417`
- `lag_02__CT_flash_duration_sum`: contribution `-0.003708`
- `lag_11__CT4__flash_duration`: contribution `-0.003549`
- `lag_07__T5__flash_duration`: contribution `+0.002788`
- `lag_02__CT1__flash_duration`: contribution `-0.002184`

### tick `134795`, seconds `44.50`, LSTM delta `+0.0955`

Top all feature movements:
- `lag_04__CT_place_ARAMP`: contribution `+0.009978`
- `lag_00__CT_kills_last_3s`: contribution `+0.006469`
- `lag_00__kill_diff_last_3s`: contribution `+0.006260`
- `lag_07__CT_place_ARAMP`: contribution `+0.006143`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004941`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `+0.003210`
- `lag_10__T3__flash_duration`: contribution `+0.002960`
- `lag_05__CT4__flash_duration`: contribution `+0.002658`
- `lag_10__T5__flash_duration`: contribution `+0.002090`
