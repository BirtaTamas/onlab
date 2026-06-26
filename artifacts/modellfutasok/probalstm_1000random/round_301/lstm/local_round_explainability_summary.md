# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `42582`, seconds `15.00`, LSTM `0.8230`, delta `+0.1431`
- tick `43158`, seconds `24.00`, LSTM `0.8374`, delta `+0.1391`
- tick `42774`, seconds `18.00`, LSTM `0.7075`, delta `-0.1286`
- tick `45686`, seconds `63.50`, LSTM `0.9540`, delta `+0.1271`
- tick `42454`, seconds `13.00`, LSTM `0.6640`, delta `+0.1246`
- tick `43510`, seconds `29.50`, LSTM `0.8708`, delta `+0.0342`
- tick `42422`, seconds `12.50`, LSTM `0.5394`, delta `+0.0309`
- tick `45110`, seconds `54.50`, LSTM `0.7949`, delta `+0.0306`
- tick `43126`, seconds `23.50`, LSTM `0.6984`, delta `-0.0278`
- tick `42806`, seconds `18.50`, LSTM `0.7334`, delta `+0.0259`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002264`, |coef| `0.002264`
- `lag_00__kill_diff_last_3s`: coefficient `0.002198`, |coef| `0.002198`
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.001744`, |coef| `0.001744`
- `lag_01__T_place_HOUSE`: coefficient `-0.001645`, |coef| `0.001645`
- `lag_00__T3__flash`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_05__CT_shots_fired_sum`: coefficient `0.001401`, |coef| `0.001401`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001394`, |coef| `0.001394`
- `lag_00__CT_damage_last_5s`: coefficient `0.001388`, |coef| `0.001388`
- `lag_00__damage_diff_last_5s`: coefficient `0.001320`, |coef| `0.001320`
- `lag_00__T_spread_xy`: coefficient `-0.001301`, |coef| `0.001301`
- `lag_13__T4__flash_duration`: coefficient `0.001299`, |coef| `0.001299`
- `lag_07__CT1__is_scoped`: coefficient `0.001284`, |coef| `0.001284`
- `lag_00__T3__utility_total`: coefficient `-0.001256`, |coef| `0.001256`
- `lag_01__T_place_ALLEY`: coefficient `0.001217`, |coef| `0.001217`
- `lag_00__T3__alive`: coefficient `-0.001203`, |coef| `0.001203`

## Top 10 utility ridge features

- `lag_00__T3__flash`: coefficient `-0.001465` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.001299` (raises CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.001256` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.001104` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001048` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.001023` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.000844` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.000784` (raises CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000756` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.000751` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002264` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002198` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.001744` (lowers CT win probability)
- `lag_01__T_place_HOUSE`: coefficient `-0.001645` (lowers CT win probability)
- `lag_05__CT_shots_fired_sum`: coefficient `0.001401` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001394` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001388` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001320` (raises CT win probability)
- `lag_00__T_spread_xy`: coefficient `-0.001301` (lowers CT win probability)
- `lag_07__CT1__is_scoped`: coefficient `0.001284` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `42582`, seconds `15.00`, LSTM delta `+0.1431`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006536`
- `lag_14__T5__flash_duration`: contribution `+0.005877`
- `lag_04__T5__flash_duration`: contribution `+0.005630`
- `lag_00__kill_diff_last_3s`: contribution `+0.005291`
- `lag_14__CT_flashed_players`: contribution `+0.004720`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `+0.005877`
- `lag_04__T5__flash_duration`: contribution `+0.005630`
- `lag_06__CT5__flash_duration`: contribution `+0.003815`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002320`

### tick `43158`, seconds `24.00`, LSTM delta `+0.1391`

Top all feature movements:
- `lag_13__T4__flash_duration`: contribution `+0.009777`
- `lag_00__T4__flash_duration`: contribution `+0.007883`
- `lag_00__CT_kills_last_3s`: contribution `+0.006536`
- `lag_00__kill_diff_last_3s`: contribution `+0.005291`
- `lag_14__CT_flashed_players`: contribution `+0.004720`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.009777`
- `lag_00__T4__flash_duration`: contribution `+0.007883`
- `lag_03__CT5__flash_duration`: contribution `+0.002725`
- `lag_00__T_flash_duration_sum`: contribution `+0.001957`
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.001918`

### tick `42774`, seconds `18.00`, LSTM delta `-0.1286`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `-0.033094`
- `lag_05__CT5__shots_fired`: contribution `-0.015547`
- `lag_00__kill_diff_last_3s`: contribution `-0.010582`
- `lag_10__T5__flash_duration`: contribution `-0.007670`
- `lag_00__CT_kills_last_3s`: contribution `-0.006536`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.007670`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.002507`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001638`

### tick `45686`, seconds `63.50`, LSTM delta `+0.1271`

Top all feature movements:
- `lag_01__T_place_HOUSE`: contribution `+0.007235`
- `lag_00__CT_kills_last_3s`: contribution `+0.006536`
- `lag_07__CT1__is_scoped`: contribution `+0.005501`
- `lag_00__kill_diff_last_3s`: contribution `+0.005291`
- `lag_01__T_place_ALLEY`: contribution `+0.005157`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.004319`
- `lag_00__T3__utility_total`: contribution `+0.003069`

### tick `42454`, seconds `13.00`, LSTM delta `+0.1246`

Top all feature movements:
- `lag_10__T5__flash_duration`: contribution `+0.007670`
- `lag_00__CT_kills_last_3s`: contribution `+0.006536`
- `lag_12__CT_place_TOPOFMID`: contribution `+0.006117`
- `lag_00__kill_diff_last_3s`: contribution `+0.005291`
- `lag_02__CT5__flash_duration`: contribution `+0.005276`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.007670`
- `lag_02__CT5__flash_duration`: contribution `+0.005276`
- `lag_00__T5__flash_duration`: contribution `+0.003943`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.002507`
- `lag_02__CT_flash_duration_sum`: contribution `+0.001982`
