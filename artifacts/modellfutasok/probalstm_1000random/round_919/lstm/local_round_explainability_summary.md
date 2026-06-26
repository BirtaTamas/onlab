# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `20`

## Largest probability jumps

- tick `203152`, seconds `19.00`, LSTM `0.3259`, delta `-0.1812`
- tick `203664`, seconds `27.00`, LSTM `0.3544`, delta `-0.1129`
- tick `204592`, seconds `41.50`, LSTM `0.1093`, delta `-0.1068`
- tick `204464`, seconds `39.50`, LSTM `0.2585`, delta `-0.0990`
- tick `204624`, seconds `42.00`, LSTM `0.0306`, delta `-0.0787`
- tick `203344`, seconds `22.00`, LSTM `0.2739`, delta `+0.0521`
- tick `203440`, seconds `23.50`, LSTM `0.3746`, delta `+0.0448`
- tick `203504`, seconds `24.50`, LSTM `0.4266`, delta `+0.0364`
- tick `203216`, seconds `20.00`, LSTM `0.2740`, delta `-0.0354`
- tick `203376`, seconds `22.50`, LSTM `0.3083`, delta `+0.0344`

## Top 15 local ridge features

- `lag_02__CT_place_LIBRARY`: coefficient `-0.001569`, |coef| `0.001569`
- `lag_04__T_place_BALCONY`: coefficient `-0.001405`, |coef| `0.001405`
- `lag_01__CT_place_LIBRARY`: coefficient `-0.001383`, |coef| `0.001383`
- `lag_05__T_place_BALCONY`: coefficient `-0.001283`, |coef| `0.001283`
- `lag_00__CT_place_BANANA`: coefficient `0.001276`, |coef| `0.001276`
- `lag_00__T_kills_last_3s`: coefficient `-0.001131`, |coef| `0.001131`
- `lag_00__CT_place_LIBRARY`: coefficient `-0.001130`, |coef| `0.001130`
- `lag_00__T_damage_last_5s`: coefficient `-0.001084`, |coef| `0.001084`
- `lag_03__CT_place_LIBRARY`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001046`, |coef| `0.001046`
- `lag_15__CT_place_BALCONY`: coefficient `0.001009`, |coef| `0.001009`
- `lag_00__damage_diff_last_5s`: coefficient `0.000991`, |coef| `0.000991`
- `lag_14__CT_place_ARCH`: coefficient `-0.000943`, |coef| `0.000943`
- `lag_00__kill_diff_last_3s`: coefficient `0.000860`, |coef| `0.000860`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000858`, |coef| `0.000858`

## Top 10 utility ridge features

- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001046` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000828` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000800` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.000773` (lowers CT win probability)
- `lag_03__CT_active_infernos`: coefficient `0.000761` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000666` (raises CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.000642` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000641` (raises CT win probability)
- `lag_14__CT_active_infernos`: coefficient `-0.000607` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000564` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_LIBRARY`: coefficient `-0.001569` (lowers CT win probability)
- `lag_04__T_place_BALCONY`: coefficient `-0.001405` (lowers CT win probability)
- `lag_01__CT_place_LIBRARY`: coefficient `-0.001383` (lowers CT win probability)
- `lag_05__T_place_BALCONY`: coefficient `-0.001283` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001276` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001131` (lowers CT win probability)
- `lag_00__CT_place_LIBRARY`: coefficient `-0.001130` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001084` (lowers CT win probability)
- `lag_03__CT_place_LIBRARY`: coefficient `-0.001063` (lowers CT win probability)
- `lag_15__CT_place_BALCONY`: coefficient `0.001009` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `203152`, seconds `19.00`, LSTM delta `-0.1812`

Top all feature movements:
- `lag_00__CT_place_BANANA`: contribution `-0.007555`
- `lag_15__CT_place_BALCONY`: contribution `-0.006476`
- `lag_14__CT_place_ARCH`: contribution `-0.003849`
- `lag_12__T4__is_scoped`: contribution `-0.003848`
- `lag_00__T_kills_last_3s`: contribution `-0.003585`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.003135`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002382`

### tick `203664`, seconds `27.00`, LSTM delta `-0.1129`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.007000`
- `lag_05__T_flashed_players`: contribution `-0.005404`
- `lag_15__CT_place_LIBRARY`: contribution `-0.004728`
- `lag_00__CT_place_BALCONY`: contribution `-0.004453`
- `lag_07__CT_place_BALCONY`: contribution `-0.003339`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.007000`
- `lag_06__CT5__flash_duration`: contribution `-0.002896`
- `lag_05__T5__flash_duration`: contribution `-0.001973`
- `lag_05__T_flash_duration_sum`: contribution `-0.001619`

### tick `204592`, seconds `41.50`, LSTM delta `-0.1068`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `-0.019320`
- `lag_01__T_place_BALCONY`: contribution `-0.009134`
- `lag_01__T_place_PIT`: contribution `-0.004598`
- `lag_04__T_place_ARCH`: contribution `-0.003692`
- `lag_00__T_shots_fired_sum`: contribution `-0.003218`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.001952`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.001937`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001537`
- `lag_06__CT_active_infernos`: contribution `-0.001477`

### tick `204464`, seconds `39.50`, LSTM delta `-0.0990`

Top all feature movements:
- `lag_02__CT_place_LIBRARY`: contribution `-0.010060`
- `lag_00__T_place_BALCONY`: contribution `-0.008103`
- `lag_00__T_place_ARCH`: contribution `-0.003701`
- `lag_09__CT4__duck_amount`: contribution `-0.002856`
- `lag_04__T2__duck_amount`: contribution `-0.002533`

Top utility-only movements:
- `lag_02__CT_active_infernos`: contribution `-0.001480`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.001418`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.001243`

### tick `204624`, seconds `42.00`, LSTM delta `-0.0787`

Top all feature movements:
- `lag_05__T_place_BALCONY`: contribution `-0.017638`
- `lag_00__T_kills_last_3s`: contribution `-0.003585`
- `lag_02__T_place_PIT`: contribution `-0.003557`
- `lag_07__CT_place_LIBRARY`: contribution `-0.003372`
- `lag_00__T_shots_fired_sum`: contribution `+0.003218`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.001941`
- `lag_03__T3__flash_duration`: contribution `-0.001734`
- `lag_03__T_A_site_active_infernos`: contribution `-0.001554`
