# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `154240`, seconds `98.50`, LSTM `0.7436`, delta `+0.3117`
- tick `152416`, seconds `70.00`, LSTM `0.8287`, delta `+0.2316`
- tick `152608`, seconds `73.00`, LSTM `0.4178`, delta `-0.1995`
- tick `154208`, seconds `98.00`, LSTM `0.4319`, delta `-0.1849`
- tick `154336`, seconds `100.00`, LSTM `0.8795`, delta `+0.1802`
- tick `152448`, seconds `70.50`, LSTM `0.6650`, delta `-0.1638`
- tick `154080`, seconds `96.00`, LSTM `0.6254`, delta `+0.1520`
- tick `148640`, seconds `11.00`, LSTM `0.8075`, delta `+0.1015`
- tick `154272`, seconds `99.00`, LSTM `0.6676`, delta `-0.0760`
- tick `154176`, seconds `97.50`, LSTM `0.6168`, delta `-0.0750`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004205`, |coef| `0.004205`
- `lag_00__CT_defusing_count`: coefficient `0.003637`, |coef| `0.003637`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003385`, |coef| `0.003385`
- `lag_00__CT_kills_last_3s`: coefficient `0.003382`, |coef| `0.003382`
- `lag_00__T_macro_B`: coefficient `-0.002630`, |coef| `0.002630`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002630`, |coef| `0.002630`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002608`, |coef| `0.002608`
- `lag_01__T_shots_fired_sum`: coefficient `0.002359`, |coef| `0.002359`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002342`, |coef| `0.002342`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002283`, |coef| `0.002283`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002281`, |coef| `0.002281`
- `lag_00__T2__shots_fired`: coefficient `-0.002230`, |coef| `0.002230`
- `lag_00__CT_velocity_mean`: coefficient `-0.002133`, |coef| `0.002133`
- `lag_01__CT_defusing_count`: coefficient `0.001997`, |coef| `0.001997`
- `lag_09__T4__is_walking`: coefficient `-0.001925`, |coef| `0.001925`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002608` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002342` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002281` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001810` (raises CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.001735` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001694` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.001437` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.001392` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001359` (lowers CT win probability)
- `lag_11__T_utility_damage_last_5s`: coefficient `-0.001299` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004205` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003637` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003385` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003382` (raises CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002630` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002630` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `0.002359` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002283` (raises CT win probability)
- `lag_00__T2__shots_fired`: coefficient `-0.002230` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002133` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `154240`, seconds `98.50`, LSTM delta `+0.3117`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015821`
- `lag_00__T_shots_fired_sum`: contribution `+0.015226`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.012792`
- `lag_00__kill_diff_last_3s`: contribution `+0.010120`
- `lag_00__CT_kills_last_3s`: contribution `+0.009766`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015821`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.012792`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.008692`

### tick `152416`, seconds `70.00`, LSTM delta `+0.2316`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.017764`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015863`
- `lag_01__T_shots_fired_sum`: contribution `+0.010613`
- `lag_06__T4__flash_duration`: contribution `+0.010181`
- `lag_00__kill_diff_last_3s`: contribution `+0.010120`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.010181`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.003875`

### tick `152608`, seconds `73.00`, LSTM delta `-0.1995`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.020240`
- `lag_05__T_bomb_zone_count`: contribution `-0.010011`
- `lag_00__CT_kills_last_3s`: contribution `-0.009766`
- `lag_13__T_bomb_zone_count`: contribution `-0.008157`
- `lag_02__CT3__is_scoped`: contribution `-0.007214`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154208`, seconds `98.00`, LSTM delta `-0.1849`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.012689`
- `lag_00__kill_diff_last_3s`: contribution `-0.010120`
- `lag_06__CT3__is_scoped`: contribution `-0.007454`
- `lag_00__T2__shots_fired`: contribution `-0.006560`
- `lag_00__T_kills_last_3s`: contribution `-0.005774`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.005418`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.004431`

### tick `154336`, seconds `100.00`, LSTM delta `+0.1802`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.035255`
- `lag_03__T_flash_alpha_mean`: contribution `+0.013840`
- `lag_00__CT_velocity_mean`: contribution `+0.007841`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.005336`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.005066`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.013840`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.005336`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.005066`
