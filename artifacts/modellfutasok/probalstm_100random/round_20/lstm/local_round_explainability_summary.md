# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `40218`, seconds `14.00`, LSTM `0.7043`, delta `+0.2396`
- tick `40474`, seconds `18.00`, LSTM `0.8350`, delta `+0.1342`
- tick `40410`, seconds `17.00`, LSTM `0.6734`, delta `-0.1067`
- tick `40826`, seconds `23.50`, LSTM `0.9425`, delta `+0.0803`
- tick `40314`, seconds `15.50`, LSTM `0.7832`, delta `+0.0481`
- tick `40570`, seconds `19.50`, LSTM `0.8728`, delta `+0.0434`
- tick `43354`, seconds `63.00`, LSTM `0.9645`, delta `+0.0395`
- tick `39930`, seconds `9.50`, LSTM `0.5515`, delta `-0.0336`
- tick `40250`, seconds `14.50`, LSTM `0.7361`, delta `+0.0317`
- tick `40186`, seconds `13.50`, LSTM `0.4647`, delta `-0.0307`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002180`, |coef| `0.002180`
- `lag_00__CT_kills_last_3s`: coefficient `0.001739`, |coef| `0.001739`
- `lag_15__CT_flashes_last_5s`: coefficient `-0.001555`, |coef| `0.001555`
- `lag_00__kill_diff_last_3s`: coefficient `0.001487`, |coef| `0.001487`
- `lag_00__damage_diff_last_5s`: coefficient `0.001384`, |coef| `0.001384`
- `lag_03__T2__shots_fired`: coefficient `0.001369`, |coef| `0.001369`
- `lag_04__T2__shots_fired`: coefficient `0.001329`, |coef| `0.001329`
- `lag_01__T2__shots_fired`: coefficient `0.001320`, |coef| `0.001320`
- `lag_00__CT_damage_last_5s`: coefficient `0.001287`, |coef| `0.001287`
- `lag_00__T2__shots_fired`: coefficient `0.001275`, |coef| `0.001275`
- `lag_02__T2__shots_fired`: coefficient `0.001270`, |coef| `0.001270`
- `lag_02__T4__flash_duration`: coefficient `0.001224`, |coef| `0.001224`
- `lag_11__T2__shots_fired`: coefficient `0.001079`, |coef| `0.001079`
- `lag_12__T2__shots_fired`: coefficient `0.001037`, |coef| `0.001037`
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001020`, |coef| `0.001020`

## Top 10 utility ridge features

- `lag_15__CT_flashes_last_5s`: coefficient `-0.001555` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001224` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001020` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000950` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000908` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000903` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000858` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000779` (lowers CT win probability)
- `lag_15__CT5__smoke`: coefficient `-0.000724` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.000703` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002180` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001739` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001487` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001384` (raises CT win probability)
- `lag_03__T2__shots_fired`: coefficient `0.001369` (raises CT win probability)
- `lag_04__T2__shots_fired`: coefficient `0.001329` (raises CT win probability)
- `lag_01__T2__shots_fired`: coefficient `0.001320` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001287` (raises CT win probability)
- `lag_00__T2__shots_fired`: coefficient `0.001275` (raises CT win probability)
- `lag_02__T2__shots_fired`: coefficient `0.001270` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `40218`, seconds `14.00`, LSTM delta `+0.2396`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.031051`
- `lag_15__CT_flashes_last_5s`: contribution `+0.017094`
- `lag_02__T4__flash_duration`: contribution `+0.008295`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.005972`
- `lag_15__T_place_LOWERMID`: contribution `+0.005474`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.017094`
- `lag_02__T4__flash_duration`: contribution `+0.008295`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.005972`
- `lag_02__T2__flash_duration`: contribution `+0.005329`
- `lag_00__T2__flash_duration`: contribution `+0.005093`

### tick `40474`, seconds `18.00`, LSTM delta `+0.1342`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.010121`
- `lag_02__T_shots_fired_sum`: contribution `+0.005852`
- `lag_00__CT_kills_last_3s`: contribution `+0.005021`
- `lag_02__CT1__flash_duration`: contribution `+0.004416`
- `lag_01__T4__shots_fired`: contribution `+0.004200`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.004416`
- `lag_08__T2__flash_duration`: contribution `+0.003945`
- `lag_04__CT1__flash_duration`: contribution `+0.003139`
- `lag_06__T5__flash_duration`: contribution `+0.002878`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.002301`

### tick `40410`, seconds `17.00`, LSTM delta `-0.1067`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.013074`
- `lag_06__T_shots_fired_sum`: contribution `-0.008696`
- `lag_00__kill_diff_last_3s`: contribution `-0.007159`
- `lag_02__CT1__flash_duration`: contribution `-0.005228`
- `lag_00__CT_kills_last_3s`: contribution `-0.005021`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.005228`
- `lag_08__T2__flash_duration`: contribution `-0.003945`
- `lag_08__T4__flash_duration`: contribution `-0.003312`
- `lag_04__T5__flash_duration`: contribution `-0.003034`
- `lag_06__CT3__flash_duration`: contribution `-0.002395`

### tick `40826`, seconds `23.50`, LSTM delta `+0.0803`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.005021`
- `lag_05__CT_shots_fired_sum`: contribution `+0.003620`
- `lag_00__kill_diff_last_3s`: contribution `+0.003580`
- `lag_00__T_shots_fired_sum`: contribution `-0.003269`
- `lag_08__T_shots_fired_sum`: contribution `-0.002663`

Top utility-only movements:
- `lag_09__T4__flash_duration`: contribution `+0.002436`
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.002079`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.001850`
- `lag_13__CT1__flash_duration`: contribution `+0.001644`
- `lag_15__CT1__flash_duration`: contribution `+0.001561`

### tick `40314`, seconds `15.50`, LSTM delta `+0.0481`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.004769`
- `lag_04__T2__shots_fired`: contribution `+0.003909`
- `lag_03__T2__shots_fired`: contribution `+0.003221`
- `lag_05__T2__shots_fired`: contribution `+0.002671`
- `lag_05__T4__flash_duration`: contribution `+0.002611`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.002611`
- `lag_01__T5__flash_duration`: contribution `+0.001575`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.001231`
- `lag_03__T2__flash_duration`: contribution `+0.001156`
