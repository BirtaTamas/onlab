# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `11251`, seconds `54.50`, LSTM `0.3373`, delta `+0.2627`
- tick `10547`, seconds `43.50`, LSTM `0.2183`, delta `-0.2395`
- tick `10483`, seconds `42.50`, LSTM `0.4332`, delta `+0.2219`
- tick `11315`, seconds `55.50`, LSTM `0.1624`, delta `-0.1489`
- tick `10227`, seconds `38.50`, LSTM `0.2621`, delta `-0.0969`
- tick `11379`, seconds `56.50`, LSTM `0.0876`, delta `-0.0746`
- tick `11411`, seconds `57.00`, LSTM `0.0249`, delta `-0.0627`
- tick `10579`, seconds `44.00`, LSTM `0.1659`, delta `-0.0524`
- tick `11187`, seconds `53.50`, LSTM `0.0679`, delta `+0.0475`
- tick `10323`, seconds `40.00`, LSTM `0.3099`, delta `+0.0473`

## Top 15 local ridge features

- `lag_05__T_place_SIDEHALL`: coefficient `-0.005118`, |coef| `0.005118`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.003811`, |coef| `0.003811`
- `lag_02__CT_flashed_players`: coefficient `0.003680`, |coef| `0.003680`
- `lag_02__T_place_SIDEHALL`: coefficient `0.003367`, |coef| `0.003367`
- `lag_00__kill_diff_last_3s`: coefficient `0.002825`, |coef| `0.002825`
- `lag_00__damage_diff_last_5s`: coefficient `0.002465`, |coef| `0.002465`
- `lag_02__T1__flash_duration`: coefficient `0.002326`, |coef| `0.002326`
- `lag_09__T5__duck_amount`: coefficient `0.002325`, |coef| `0.002325`
- `lag_00__CT_kills_last_3s`: coefficient `0.002236`, |coef| `0.002236`
- `lag_04__T_place_SIDEHALL`: coefficient `-0.002217`, |coef| `0.002217`
- `lag_05__T_bomb_zone_count`: coefficient `-0.002066`, |coef| `0.002066`
- `lag_11__T_bomb_zone_count`: coefficient `0.002056`, |coef| `0.002056`
- `lag_10__T3__duck_amount`: coefficient `-0.001997`, |coef| `0.001997`
- `lag_08__T_place_MAINHALL`: coefficient `-0.001932`, |coef| `0.001932`
- `lag_13__T2__is_walking`: coefficient `-0.001907`, |coef| `0.001907`

## Top 10 utility ridge features

- `lag_02__T1__flash_duration`: coefficient `0.002326` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001553` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.001541` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001141` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.000971` (lowers CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000966` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000853` (raises CT win probability)
- `lag_05__CT3__flash`: coefficient `0.000774` (raises CT win probability)
- `lag_10__CT1__smoke`: coefficient `0.000773` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000763` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_SIDEHALL`: coefficient `-0.005118` (lowers CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.003811` (lowers CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.003680` (raises CT win probability)
- `lag_02__T_place_SIDEHALL`: coefficient `0.003367` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002825` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002465` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.002325` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002236` (raises CT win probability)
- `lag_04__T_place_SIDEHALL`: coefficient `-0.002217` (lowers CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.002066` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `11251`, seconds `54.50`, LSTM delta `+0.2627`

Top all feature movements:
- `lag_05__T_place_SIDEHALL`: contribution `+0.033173`
- `lag_02__CT_flashed_players`: contribution `+0.032232`
- `lag_02__T1__flash_duration`: contribution `+0.016339`
- `lag_05__T_bomb_zone_count`: contribution `+0.012029`
- `lag_11__T_bomb_zone_count`: contribution `+0.011970`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.016339`
- `lag_02__T_flash_duration_sum`: contribution `+0.006240`
- `lag_02__T2__flash_duration`: contribution `+0.004121`
- `lag_02__CT_flash_duration_sum`: contribution `+0.003149`

### tick `10547`, seconds `43.50`, LSTM delta `-0.2395`

Top all feature movements:
- `lag_05__T_place_SIDEHALL`: contribution `-0.033173`
- `lag_02__T_place_SIDEHALL`: contribution `-0.021821`
- `lag_04__T_place_SIDEHALL`: contribution `-0.014368`
- `lag_09__T5__duck_amount`: contribution `-0.008829`
- `lag_02__CT_place_ALLEY`: contribution `-0.008316`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10483`, seconds `42.50`, LSTM delta `+0.2219`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.024697`
- `lag_02__T_place_SIDEHALL`: contribution `+0.021821`
- `lag_09__T5__duck_amount`: contribution `+0.008829`
- `lag_00__CT_place_ALLEY`: contribution `+0.008772`
- `lag_03__T_place_SIDEHALL`: contribution `+0.007637`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11315`, seconds `55.50`, LSTM delta `-0.1489`

Top all feature movements:
- `lag_01__CT_flashed_players`: contribution `-0.011325`
- `lag_04__CT_flashed_players`: contribution `-0.008714`
- `lag_02__CT_flashed_players`: contribution `-0.008058`
- `lag_04__T1__flash_duration`: contribution `-0.006818`
- `lag_00__kill_diff_last_3s`: contribution `-0.006800`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.006818`
- `lag_02__T2__flash_duration`: contribution `-0.004121`

### tick `10227`, seconds `38.50`, LSTM delta `-0.0969`

Top all feature movements:
- `lag_14__T1__duck_amount`: contribution `-0.007050`
- `lag_09__T_place_MAINHALL`: contribution `-0.006724`
- `lag_14__T3__duck_amount`: contribution `+0.005697`
- `lag_01__T5__duck_amount`: contribution `-0.005641`
- `lag_03__T3__duck_amount`: contribution `-0.004456`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `-0.002093`
