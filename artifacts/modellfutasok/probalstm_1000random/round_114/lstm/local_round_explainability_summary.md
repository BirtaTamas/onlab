# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `69898`, seconds `63.50`, LSTM `0.5335`, delta `+0.3533`
- tick `67178`, seconds `21.00`, LSTM `0.2961`, delta `-0.2444`
- tick `71690`, seconds `91.50`, LSTM `0.9058`, delta `+0.1870`
- tick `69866`, seconds `63.00`, LSTM `0.1802`, delta `-0.1653`
- tick `71146`, seconds `83.00`, LSTM `0.6494`, delta `+0.1407`
- tick `71530`, seconds `89.00`, LSTM `0.6851`, delta `+0.0767`
- tick `67146`, seconds `20.50`, LSTM `0.5404`, delta `-0.0602`
- tick `67210`, seconds `21.50`, LSTM `0.2483`, delta `-0.0478`
- tick `71434`, seconds `87.50`, LSTM `0.6351`, delta `-0.0445`
- tick `67242`, seconds `22.00`, LSTM `0.2047`, delta `-0.0436`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004262`, |coef| `0.004262`
- `lag_00__CT_kills_last_3s`: coefficient `0.004134`, |coef| `0.004134`
- `lag_00__damage_diff_last_5s`: coefficient `0.003917`, |coef| `0.003917`
- `lag_00__CT_damage_last_5s`: coefficient `0.002945`, |coef| `0.002945`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002614`, |coef| `0.002614`
- `lag_00__CT2__is_scoped`: coefficient `-0.002138`, |coef| `0.002138`
- `lag_15__CT5__duck_amount`: coefficient `-0.001988`, |coef| `0.001988`
- `lag_05__CT3__flash_duration`: coefficient `0.001966`, |coef| `0.001966`
- `lag_00__CT5__shots_fired`: coefficient `0.001879`, |coef| `0.001879`
- `lag_05__CT5__duck_amount`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001815`, |coef| `0.001815`
- `lag_15__CT_place_BALCONY`: coefficient `0.001773`, |coef| `0.001773`
- `lag_13__CT4__is_walking`: coefficient `-0.001737`, |coef| `0.001737`
- `lag_15__CT_place_LIBRARY`: coefficient `0.001678`, |coef| `0.001678`
- `lag_05__T4__is_walking`: coefficient `0.001669`, |coef| `0.001669`

## Top 10 utility ridge features

- `lag_05__CT3__flash_duration`: coefficient `0.001966` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001815` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.001452` (raises CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.001394` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.001368` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001346` (raises CT win probability)
- `lag_00__T3__molly`: coefficient `-0.001270` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.001239` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.001233` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.001174` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004262` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004134` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003917` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002945` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002614` (raises CT win probability)
- `lag_00__CT2__is_scoped`: coefficient `-0.002138` (lowers CT win probability)
- `lag_15__CT5__duck_amount`: coefficient `-0.001988` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.001879` (raises CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `-0.001816` (lowers CT win probability)
- `lag_15__CT_place_BALCONY`: coefficient `0.001773` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `69898`, seconds `63.50`, LSTM delta `+0.3533`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.023870`
- `lag_00__kill_diff_last_3s`: contribution `+0.020515`
- `lag_00__CT2__is_scoped`: contribution `+0.013084`
- `lag_00__damage_diff_last_5s`: contribution `+0.012636`
- `lag_00__CT_damage_last_5s`: contribution `+0.009181`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67178`, seconds `21.00`, LSTM delta `-0.2444`

Top all feature movements:
- `lag_09__T_place_BALCONY`: contribution `-0.019194`
- `lag_08__T_place_BALCONY`: contribution `-0.018973`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.012435`
- `lag_00__kill_diff_last_3s`: contribution `-0.010257`
- `lag_00__CT1__flash_duration`: contribution `-0.008603`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.012435`
- `lag_00__CT1__flash_duration`: contribution `-0.008603`
- `lag_02__CT1__flash_duration`: contribution `-0.008436`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003827`
- `lag_07__T_A_site_active_infernos`: contribution `-0.003493`

### tick `71690`, seconds `91.50`, LSTM delta `+0.1870`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014528`
- `lag_05__CT3__flash_duration`: contribution `+0.013272`
- `lag_00__CT_kills_last_3s`: contribution `+0.011935`
- `lag_00__kill_diff_last_3s`: contribution `+0.010257`
- `lag_08__CT5__flash_duration`: contribution `+0.009336`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.013272`
- `lag_08__CT5__flash_duration`: contribution `+0.009336`
- `lag_05__T4__flash_duration`: contribution `+0.006885`
- `lag_05__T1__flash_duration`: contribution `+0.005710`
- `lag_05__T_flash_duration_sum`: contribution `+0.003613`

### tick `69866`, seconds `63.00`, LSTM delta `-0.1653`

Top all feature movements:
- `lag_15__CT_place_BALCONY`: contribution `-0.011380`
- `lag_00__kill_diff_last_3s`: contribution `-0.010257`
- `lag_00__damage_diff_last_5s`: contribution `-0.008836`
- `lag_15__CT5__duck_amount`: contribution `-0.007504`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.004483`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71146`, seconds `83.00`, LSTM delta `+0.1407`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `+0.013084`
- `lag_00__CT_kills_last_3s`: contribution `+0.011935`
- `lag_15__CT_place_LIBRARY`: contribution `+0.010756`
- `lag_00__kill_diff_last_3s`: contribution `+0.010257`
- `lag_00__damage_diff_last_5s`: contribution `+0.008836`

Top utility-only movements:
- No utility movement among the top local contributors.
