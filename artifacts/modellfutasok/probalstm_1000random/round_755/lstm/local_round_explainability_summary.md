# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `12622`, seconds `73.50`, LSTM `0.8138`, delta `+0.2380`
- tick `12494`, seconds `71.50`, LSTM `0.7201`, delta `-0.1977`
- tick `12046`, seconds `64.50`, LSTM `0.7771`, delta `-0.1657`
- tick `12590`, seconds `73.00`, LSTM `0.5758`, delta `-0.1161`
- tick `12750`, seconds `75.50`, LSTM `0.9099`, delta `+0.1157`
- tick `12238`, seconds `67.50`, LSTM `0.7918`, delta `+0.1023`
- tick `12270`, seconds `68.00`, LSTM `0.8797`, delta `+0.0879`
- tick `8878`, seconds `15.00`, LSTM `0.9123`, delta `+0.0450`
- tick `12526`, seconds `72.00`, LSTM `0.6765`, delta `-0.0436`
- tick `12686`, seconds `74.50`, LSTM `0.8349`, delta `+0.0433`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002254`, |coef| `0.002254`
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_00__T_kills_last_3s`: coefficient `-0.001709`, |coef| `0.001709`
- `lag_08__CT_place_HOLE`: coefficient `-0.001687`, |coef| `0.001687`
- `lag_00__damage_diff_last_5s`: coefficient `0.001680`, |coef| `0.001680`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001447`, |coef| `0.001447`
- `lag_11__T2__flash_duration`: coefficient `-0.001304`, |coef| `0.001304`
- `lag_00__T4__flash_duration`: coefficient `0.001272`, |coef| `0.001272`
- `lag_06__CT_place_HOLE`: coefficient `-0.001258`, |coef| `0.001258`
- `lag_04__CT_place_HOLE`: coefficient `0.001210`, |coef| `0.001210`
- `lag_15__T1__flash_duration`: coefficient `0.001179`, |coef| `0.001179`
- `lag_00__CT_kills_last_3s`: coefficient `0.001147`, |coef| `0.001147`
- `lag_01__kill_diff_last_3s`: coefficient `0.001135`, |coef| `0.001135`
- `lag_08__T_flashed_players`: coefficient `-0.001113`, |coef| `0.001113`
- `lag_00__T_flash_duration_sum`: coefficient `0.001109`, |coef| `0.001109`

## Top 10 utility ridge features

- `lag_11__T2__flash_duration`: coefficient `-0.001304` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001272` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.001179` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.001109` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001016` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.000992` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.000951` (raises CT win probability)
- `lag_12__T2__flash_duration`: coefficient `0.000932` (raises CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `-0.000820` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.000734` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002254` (raises CT win probability)
- `lag_03__T_place_EXTENDEDA`: coefficient `-0.001876` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001709` (lowers CT win probability)
- `lag_08__CT_place_HOLE`: coefficient `-0.001687` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001680` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001447` (lowers CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `-0.001258` (lowers CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `0.001210` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001147` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.001135` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `12622`, seconds `73.50`, LSTM delta `+0.2380`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `+0.018835`
- `lag_10__CT_place_HOLE`: contribution `+0.010249`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.009302`
- `lag_11__T2__flash_duration`: contribution `+0.008307`
- `lag_12__T_flash_duration_sum`: contribution `+0.008277`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.008307`
- `lag_12__T_flash_duration_sum`: contribution `+0.008277`
- `lag_12__T2__flash_duration`: contribution `+0.005937`
- `lag_12__T4__flash_duration`: contribution `+0.004029`
- `lag_12__T1__flash_duration`: contribution `+0.003470`

### tick `12494`, seconds `71.50`, LSTM delta `-0.1977`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `-0.014045`
- `lag_04__CT_place_HOLE`: contribution `-0.013505`
- `lag_08__T_flash_duration_sum`: contribution `-0.008639`
- `lag_08__T_flashed_players`: contribution `-0.008594`
- `lag_00__kill_diff_last_3s`: contribution `-0.005426`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `-0.008639`
- `lag_07__T2__flash_duration`: contribution `-0.004673`
- `lag_08__T4__flash_duration`: contribution `-0.004548`
- `lag_08__T1__flash_duration`: contribution `-0.003136`
- `lag_08__T2__flash_duration`: contribution `-0.002860`

### tick `12046`, seconds `64.50`, LSTM delta `-0.1657`

Top all feature movements:
- `lag_03__T_place_EXTENDEDA`: contribution `-0.009302`
- `lag_15__T1__flash_duration`: contribution `-0.008612`
- `lag_00__T_place_EXTENDEDA`: contribution `-0.007173`
- `lag_00__kill_diff_last_3s`: contribution `-0.005426`
- `lag_00__T_kills_last_3s`: contribution `-0.005414`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `-0.008612`
- `lag_00__CT5__flash`: contribution `-0.002468`

### tick `12590`, seconds `73.00`, LSTM delta `-0.1161`

Top all feature movements:
- `lag_11__T2__flash_duration`: contribution `-0.008307`
- `lag_00__T4__flash_duration`: contribution `-0.008098`
- `lag_11__T_flash_duration_sum`: contribution `-0.007138`
- `lag_00__T1__flash_duration`: contribution `-0.006159`
- `lag_00__T_flash_duration_sum`: contribution `-0.005668`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.008307`
- `lag_00__T4__flash_duration`: contribution `-0.008098`
- `lag_11__T_flash_duration_sum`: contribution `-0.007138`
- `lag_00__T1__flash_duration`: contribution `-0.006159`
- `lag_00__T_flash_duration_sum`: contribution `-0.005668`

### tick `12750`, seconds `75.50`, LSTM delta `+0.1157`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `+0.010102`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.007173`
- `lag_00__kill_diff_last_3s`: contribution `+0.005426`
- `lag_12__CT_place_HOLE`: contribution `+0.005376`
- `lag_13__T1__is_scoped`: contribution `-0.004898`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `+0.004266`
- `lag_05__T1__flash_duration`: contribution `+0.002699`
- `lag_05__T4__flash_duration`: contribution `+0.002359`
