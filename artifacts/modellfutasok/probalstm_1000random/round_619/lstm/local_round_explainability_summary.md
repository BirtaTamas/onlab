# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `83945`, seconds `46.00`, LSTM `0.8923`, delta `+0.1864`
- tick `86569`, seconds `87.00`, LSTM `0.7741`, delta `-0.1594`
- tick `83689`, seconds `42.00`, LSTM `0.7138`, delta `+0.1487`
- tick `84649`, seconds `57.00`, LSTM `0.8185`, delta `-0.1192`
- tick `85289`, seconds `67.00`, LSTM `0.9428`, delta `+0.0902`
- tick `85929`, seconds `77.00`, LSTM `0.8934`, delta `-0.0725`
- tick `83913`, seconds `45.50`, LSTM `0.7059`, delta `+0.0722`
- tick `84777`, seconds `59.00`, LSTM `0.8489`, delta `+0.0506`
- tick `84969`, seconds `62.00`, LSTM `0.8236`, delta `+0.0382`
- tick `84873`, seconds `60.50`, LSTM `0.7888`, delta `-0.0373`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002964`, |coef| `0.002964`
- `lag_00__T_kills_last_3s`: coefficient `-0.002752`, |coef| `0.002752`
- `lag_14__T_kills_last_3s`: coefficient `0.002377`, |coef| `0.002377`
- `lag_00__CT_burning_players`: coefficient `0.002286`, |coef| `0.002286`
- `lag_00__CT3__duck_amount`: coefficient `0.001989`, |coef| `0.001989`
- `lag_00__CT3__alive`: coefficient `0.001946`, |coef| `0.001946`
- `lag_10__T_damage_last_5s`: coefficient `0.001872`, |coef| `0.001872`
- `lag_14__CT_place_OBSERVATION`: coefficient `0.001777`, |coef| `0.001777`
- `lag_09__CT_place_OBSERVATION`: coefficient `-0.001749`, |coef| `0.001749`
- `lag_00__CT3__armor`: coefficient `0.001696`, |coef| `0.001696`
- `lag_08__T_duck_amount_mean`: coefficient `0.001675`, |coef| `0.001675`
- `lag_10__T3__is_scoped`: coefficient `0.001636`, |coef| `0.001636`
- `lag_00__CT3__has_helmet`: coefficient `0.001619`, |coef| `0.001619`
- `lag_00__damage_diff_last_5s`: coefficient `0.001576`, |coef| `0.001576`
- `lag_09__T_duck_amount_mean`: coefficient `-0.001572`, |coef| `0.001572`

## Top 10 utility ridge features

- `lag_04__T3__flash_duration`: coefficient `-0.001211` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.000604` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.000561` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000511` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000501` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000335` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.000326` (raises CT win probability)
- `lag_14__T1__smoke`: coefficient `-0.000324` (lowers CT win probability)
- `lag_06__T1__utility_total`: coefficient `-0.000323` (lowers CT win probability)
- `lag_06__T1__smoke`: coefficient `-0.000318` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002964` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002752` (lowers CT win probability)
- `lag_14__T_kills_last_3s`: coefficient `0.002377` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.002286` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001989` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001946` (raises CT win probability)
- `lag_10__T_damage_last_5s`: coefficient `0.001872` (raises CT win probability)
- `lag_14__CT_place_OBSERVATION`: coefficient `0.001777` (raises CT win probability)
- `lag_09__CT_place_OBSERVATION`: coefficient `-0.001749` (lowers CT win probability)
- `lag_00__CT3__armor`: coefficient `0.001696` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `83945`, seconds `46.00`, LSTM delta `+0.1864`

Top all feature movements:
- `lag_14__CT_place_OBSERVATION`: contribution `+0.030938`
- `lag_09__CT_place_OBSERVATION`: contribution `+0.030450`
- `lag_01__T_place_GARAGE`: contribution `+0.012460`
- `lag_10__T3__is_scoped`: contribution `+0.010496`
- `lag_00__T_place_GARAGE`: contribution `+0.008455`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86569`, seconds `87.00`, LSTM delta `-0.1594`

Top all feature movements:
- `lag_08__T_duck_amount_mean`: contribution `-0.009741`
- `lag_09__T_duck_amount_mean`: contribution `-0.009140`
- `lag_00__T_kills_last_3s`: contribution `-0.008720`
- `lag_14__T_kills_last_3s`: contribution `-0.007530`
- `lag_00__CT3__duck_amount`: contribution `-0.007401`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83689`, seconds `42.00`, LSTM delta `+0.1487`

Top all feature movements:
- `lag_06__CT_place_OBSERVATION`: contribution `+0.023895`
- `lag_01__CT_place_OBSERVATION`: contribution `+0.019983`
- `lag_09__CT_place_CRANE`: contribution `+0.009530`
- `lag_08__CT_place_CRANE`: contribution `+0.007950`
- `lag_00__kill_diff_last_3s`: contribution `+0.007133`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84649`, seconds `57.00`, LSTM delta `-0.1192`

Top all feature movements:
- `lag_03__T_place_MINI`: contribution `-0.015952`
- `lag_10__T3__is_scoped`: contribution `-0.010496`
- `lag_00__T_kills_last_3s`: contribution `-0.008720`
- `lag_00__CT_place_HUTROOF`: contribution `-0.007568`
- `lag_14__T_kills_last_3s`: contribution `-0.007530`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `-0.002958`

### tick `85289`, seconds `67.00`, LSTM delta `+0.0902`

Top all feature movements:
- `lag_13__CT_place_SECRET`: contribution `+0.014020`
- `lag_04__T3__flash_duration`: contribution `+0.008209`
- `lag_14__T_kills_last_3s`: contribution `-0.007530`
- `lag_00__kill_diff_last_3s`: contribution `+0.007133`
- `lag_04__T3__is_scoped`: contribution `+0.005524`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.008209`
