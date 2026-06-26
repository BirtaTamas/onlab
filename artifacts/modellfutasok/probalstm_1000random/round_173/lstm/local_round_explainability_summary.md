# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `19`

## Largest probability jumps

- tick `147479`, seconds `68.00`, LSTM `0.8157`, delta `+0.1294`
- tick `147223`, seconds `64.00`, LSTM `0.8047`, delta `+0.1157`
- tick `148535`, seconds `84.50`, LSTM `0.8812`, delta `+0.1115`
- tick `147351`, seconds `66.00`, LSTM `0.7649`, delta `-0.0853`
- tick `147287`, seconds `65.00`, LSTM `0.8876`, delta `+0.0671`
- tick `148567`, seconds `85.00`, LSTM `0.9445`, delta `+0.0633`
- tick `148439`, seconds `83.00`, LSTM `0.7924`, delta `-0.0546`
- tick `146935`, seconds `59.50`, LSTM `0.6469`, delta `-0.0529`
- tick `147415`, seconds `67.00`, LSTM `0.6989`, delta `-0.0474`
- tick `147159`, seconds `63.00`, LSTM `0.6775`, delta `-0.0415`

## Top 15 local ridge features

- `lag_05__CT_shots_fired_sum`: coefficient `-0.001586`, |coef| `0.001586`
- `lag_00__kill_diff_last_3s`: coefficient `0.001572`, |coef| `0.001572`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001402`, |coef| `0.001402`
- `lag_00__CT_kills_last_3s`: coefficient `0.001254`, |coef| `0.001254`
- `lag_03__T5__flash_duration`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_05__CT2__shots_fired`: coefficient `-0.001031`, |coef| `0.001031`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000954`, |coef| `0.000954`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.000908`, |coef| `0.000908`
- `lag_00__T_macro_B`: coefficient `-0.000908`, |coef| `0.000908`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000871`, |coef| `0.000871`
- `lag_01__CT2__shots_fired`: coefficient `0.000842`, |coef| `0.000842`
- `lag_03__CT_place_RUINS`: coefficient `-0.000830`, |coef| `0.000830`
- `lag_08__T3__flash_duration`: coefficient `0.000823`, |coef| `0.000823`
- `lag_04__T5__flash_duration`: coefficient `-0.000823`, |coef| `0.000823`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000799`, |coef| `0.000799`

## Top 10 utility ridge features

- `lag_03__T5__flash_duration`: coefficient `-0.001128` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000954` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.000823` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.000823` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.000696` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000685` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.000643` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000638` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000630` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000549` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_shots_fired_sum`: coefficient `-0.001586` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001572` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001402` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001254` (raises CT win probability)
- `lag_05__CT2__shots_fired`: coefficient `-0.001031` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.000908` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.000908` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000871` (raises CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `0.000842` (raises CT win probability)
- `lag_03__CT_place_RUINS`: coefficient `-0.000830` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `147479`, seconds `68.00`, LSTM delta `+0.1294`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `+0.034160`
- `lag_05__CT2__shots_fired`: contribution `+0.011275`
- `lag_13__CT_place_LIBRARY`: contribution `+0.003107`
- `lag_02__T_velocity_mean`: contribution `+0.002364`
- `lag_06__CT_shots_fired_sum`: contribution `+0.002257`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.001908`
- `lag_04__CT2__flash_duration`: contribution `+0.001870`
- `lag_06__T2__flash_duration`: contribution `+0.001643`

### tick `147223`, seconds `64.00`, LSTM delta `+0.1157`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006819`
- `lag_08__T3__flash_duration`: contribution `+0.005990`
- `lag_09__CT2__flash_duration`: contribution `+0.005191`
- `lag_00__T4__flash_duration`: contribution `+0.004465`
- `lag_00__kill_diff_last_3s`: contribution `+0.003783`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `+0.005990`
- `lag_09__CT2__flash_duration`: contribution `+0.005191`
- `lag_00__T4__flash_duration`: contribution `+0.004465`
- `lag_08__T_flash_duration_sum`: contribution `+0.002287`
- `lag_08__T4__flash_duration`: contribution `+0.001875`

### tick `148535`, seconds `84.50`, LSTM delta `+0.1115`

Top all feature movements:
- `lag_03__T5__flash_duration`: contribution `+0.007973`
- `lag_00__kill_diff_last_3s`: contribution `+0.003783`
- `lag_00__CT_kills_last_3s`: contribution `+0.003621`
- `lag_07__CT3__duck_amount`: contribution `+0.002919`
- `lag_03__CT_place_RUINS`: contribution `+0.002900`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.007973`
- `lag_03__T_flash_duration_sum`: contribution `+0.001449`

### tick `147351`, seconds `66.00`, LSTM delta `-0.0853`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.018761`
- `lag_01__CT2__shots_fired`: contribution `-0.009206`
- `lag_05__CT_shots_fired_sum`: contribution `-0.005510`
- `lag_00__kill_diff_last_3s`: contribution `-0.003783`
- `lag_12__T3__flash_duration`: contribution `-0.003676`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `-0.003676`
- `lag_13__CT2__flash_duration`: contribution `-0.002466`
- `lag_12__T4__flash_duration`: contribution `-0.002230`
- `lag_04__T4__flash_duration`: contribution `-0.001978`
- `lag_12__T_flash_duration_sum`: contribution `-0.001947`

### tick `147287`, seconds `65.00`, LSTM delta `+0.0671`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.011689`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004236`
- `lag_00__kill_diff_last_3s`: contribution `+0.003783`
- `lag_00__CT_kills_last_3s`: contribution `+0.003621`
- `lag_11__CT_place_LIBRARY`: contribution `+0.002893`

Top utility-only movements:
- No utility movement among the top local contributors.
