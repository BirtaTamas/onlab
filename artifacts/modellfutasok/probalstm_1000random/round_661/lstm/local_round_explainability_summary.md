# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `164982`, seconds `93.00`, LSTM `0.8079`, delta `+0.2836`
- tick `164918`, seconds `92.00`, LSTM `0.5159`, delta `-0.2622`
- tick `164598`, seconds `87.00`, LSTM `0.8971`, delta `+0.2440`
- tick `164278`, seconds `82.00`, LSTM `0.8278`, delta `+0.2104`
- tick `164758`, seconds `89.50`, LSTM `0.7832`, delta `-0.1550`
- tick `164438`, seconds `84.50`, LSTM `0.5729`, delta `-0.1211`
- tick `164310`, seconds `82.50`, LSTM `0.7189`, delta `-0.1089`
- tick `164150`, seconds `80.00`, LSTM `0.5458`, delta `+0.0834`
- tick `164822`, seconds `90.50`, LSTM `0.7083`, delta `-0.0825`
- tick `164854`, seconds `91.00`, LSTM `0.7848`, delta `+0.0766`

## Top 15 local ridge features

- `lag_06__T_shots_fired_sum`: coefficient `-0.002469`, |coef| `0.002469`
- `lag_00__kill_diff_last_3s`: coefficient `0.002181`, |coef| `0.002181`
- `lag_00__CT_kills_last_3s`: coefficient `0.001825`, |coef| `0.001825`
- `lag_04__T_shots_fired_sum`: coefficient `0.001699`, |coef| `0.001699`
- `lag_15__CT_shots_fired_sum`: coefficient `0.001628`, |coef| `0.001628`
- `lag_04__T_velocity_mean`: coefficient `-0.001605`, |coef| `0.001605`
- `lag_12__CT1__duck_amount`: coefficient `-0.001476`, |coef| `0.001476`
- `lag_05__T_place_PIT`: coefficient `-0.001424`, |coef| `0.001424`
- `lag_12__T_place_QUAD`: coefficient `0.001411`, |coef| `0.001411`
- `lag_12__CT1__shots_fired`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_06__T5__shots_fired`: coefficient `-0.001394`, |coef| `0.001394`
- `lag_04__T2__duck_amount`: coefficient `-0.001374`, |coef| `0.001374`
- `lag_14__CT_shots_fired_sum`: coefficient `0.001368`, |coef| `0.001368`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001315`, |coef| `0.001315`
- `lag_00__damage_diff_last_5s`: coefficient `0.001306`, |coef| `0.001306`

## Top 10 utility ridge features

- `lag_15__T4__flash_duration`: coefficient `0.001092` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000927` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.000919` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.000816` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000759` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000682` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000662` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000609` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.000604` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.000589` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T_shots_fired_sum`: coefficient `-0.002469` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002181` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001825` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `0.001699` (raises CT win probability)
- `lag_15__CT_shots_fired_sum`: coefficient `0.001628` (raises CT win probability)
- `lag_04__T_velocity_mean`: coefficient `-0.001605` (lowers CT win probability)
- `lag_12__CT1__duck_amount`: coefficient `-0.001476` (lowers CT win probability)
- `lag_05__T_place_PIT`: coefficient `-0.001424` (lowers CT win probability)
- `lag_12__T_place_QUAD`: coefficient `0.001411` (raises CT win probability)
- `lag_12__CT1__shots_fired`: coefficient `-0.001401` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `164982`, seconds `93.00`, LSTM delta `+0.2836`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.035176`
- `lag_04__T_velocity_mean`: contribution `+0.009245`
- `lag_05__T_place_PIT`: contribution `+0.008989`
- `lag_06__T5__shots_fired`: contribution `+0.008570`
- `lag_10__T_place_PIT`: contribution `+0.007933`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `164918`, seconds `92.00`, LSTM delta `-0.2622`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.024197`
- `lag_05__T_shots_fired_sum`: contribution `-0.009132`
- `lag_06__T_shots_fired_sum`: contribution `-0.007406`
- `lag_03__T_velocity_mean`: contribution `-0.006333`
- `lag_08__T_place_PIT`: contribution `-0.006250`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `-0.003913`

### tick `164598`, seconds `87.00`, LSTM delta `+0.2440`

Top all feature movements:
- `lag_14__T_place_BALCONY`: contribution `+0.014237`
- `lag_12__CT1__shots_fired`: contribution `+0.012584`
- `lag_15__T_place_BALCONY`: contribution `+0.012217`
- `lag_12__CT_shots_fired_sum`: contribution `+0.011255`
- `lag_06__T_shots_fired_sum`: contribution `-0.009257`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.003863`

### tick `164278`, seconds `82.00`, LSTM delta `+0.2104`

Top all feature movements:
- `lag_12__T_place_QUAD`: contribution `+0.033983`
- `lag_08__T_place_QUAD`: contribution `+0.023220`
- `lag_05__T_place_BALCONY`: contribution `+0.011608`
- `lag_04__T_place_BALCONY`: contribution `+0.008286`
- `lag_06__T_flashed_players`: contribution `+0.007755`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `+0.005865`
- `lag_06__T1__flash_duration`: contribution `+0.003862`

### tick `164758`, seconds `89.50`, LSTM delta `-0.1550`

Top all feature movements:
- `lag_15__CT_shots_fired_sum`: contribution `-0.010181`
- `lag_13__CT_place_LIBRARY`: contribution `-0.008100`
- `lag_00__T_shots_fired_sum`: contribution `-0.007949`
- `lag_09__CT_place_LIBRARY`: contribution `-0.006729`
- `lag_06__T_shots_fired_sum`: contribution `-0.005554`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `-0.003928`
- `lag_14__CT3__flash_duration`: contribution `-0.003134`
