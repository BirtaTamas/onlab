# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `4`

## Largest probability jumps

- tick `25984`, seconds `59.00`, LSTM `0.9164`, delta `+0.2001`
- tick `25632`, seconds `53.50`, LSTM `0.5559`, delta `+0.1458`
- tick `25792`, seconds `56.00`, LSTM `0.7831`, delta `+0.1455`
- tick `25728`, seconds `55.00`, LSTM `0.6952`, delta `+0.1147`
- tick `25216`, seconds `47.00`, LSTM `0.3672`, delta `-0.1053`
- tick `25024`, seconds `44.00`, LSTM `0.5401`, delta `-0.0942`
- tick `25184`, seconds `46.50`, LSTM `0.4725`, delta `-0.0776`
- tick `24544`, seconds `36.50`, LSTM `0.6224`, delta `-0.0623`
- tick `25760`, seconds `55.50`, LSTM `0.6375`, delta `-0.0577`
- tick `25312`, seconds `48.50`, LSTM `0.3860`, delta `+0.0528`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002269`, |coef| `0.002269`
- `lag_06__CT_shots_fired_sum`: coefficient `-0.002263`, |coef| `0.002263`
- `lag_07__T_bomb_zone_count`: coefficient `-0.002206`, |coef| `0.002206`
- `lag_00__CT_kills_last_3s`: coefficient `0.001753`, |coef| `0.001753`
- `lag_14__T_bomb_zone_count`: coefficient `0.001746`, |coef| `0.001746`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001704`, |coef| `0.001704`
- `lag_00__damage_diff_last_5s`: coefficient `0.001701`, |coef| `0.001701`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001589`, |coef| `0.001589`
- `lag_00__T_macro_B`: coefficient `-0.001589`, |coef| `0.001589`
- `lag_06__T_bomb_zone_count`: coefficient `0.001531`, |coef| `0.001531`
- `lag_01__T_bomb_zone_count`: coefficient `-0.001516`, |coef| `0.001516`
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.001503`, |coef| `0.001503`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001502`, |coef| `0.001502`
- `lag_03__T_bomb_zone_count`: coefficient `0.001443`, |coef| `0.001443`
- `lag_11__kill_diff_last_3s`: coefficient `0.001405`, |coef| `0.001405`

## Top 10 utility ridge features

- `lag_03__CT_utility_damage_last_5s`: coefficient `0.001503` (raises CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.001283` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.001236` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.001230` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.001149` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `0.001095` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.001051` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.001034` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001008` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000968` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002269` (raises CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `-0.002263` (lowers CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.002206` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001753` (raises CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `0.001746` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001704` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001701` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001589` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001589` (lowers CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `0.001531` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25984`, seconds `59.00`, LSTM delta `+0.2001`

Top all feature movements:
- `lag_06__CT_shots_fired_sum`: contribution `+0.014152`
- `lag_07__T_bomb_zone_count`: contribution `+0.012844`
- `lag_14__T_bomb_zone_count`: contribution `+0.010165`
- `lag_06__CT4__shots_fired`: contribution `+0.006036`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.005415`

Top utility-only movements:
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.005415`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.005058`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.003642`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.003390`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.003327`

### tick `25632`, seconds `53.50`, LSTM delta `+0.1458`

Top all feature movements:
- `lag_13__CT_place_ENTRANCE`: contribution `+0.011349`
- `lag_14__T_shots_fired_sum`: contribution `+0.008795`
- `lag_03__T_bomb_zone_count`: contribution `+0.008403`
- `lag_14__T3__shots_fired`: contribution `+0.006438`
- `lag_00__kill_diff_last_3s`: contribution `+0.005460`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.002926`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.002199`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.002074`

### tick `25792`, seconds `56.00`, LSTM delta `+0.1455`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `+0.008823`
- `lag_04__CT_shots_fired_sum`: contribution `+0.005921`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.005651`
- `lag_00__kill_diff_last_3s`: contribution `+0.005460`
- `lag_08__T_bomb_zone_count`: contribution `+0.005427`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.005651`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.003796`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.002637`

### tick `25728`, seconds `55.00`, LSTM delta `+0.1147`

Top all feature movements:
- `lag_06__T_bomb_zone_count`: contribution `+0.008914`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.006617`
- `lag_03__CT_shots_fired_sum`: contribution `+0.005216`
- `lag_07__T5__duck_amount`: contribution `+0.005099`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.004462`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.006617`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.004462`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.003761`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.003649`

### tick `25216`, seconds `47.00`, LSTM delta `-0.1053`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.010432`
- `lag_03__CT1__shots_fired`: contribution `-0.006887`
- `lag_04__CT_shots_fired_sum`: contribution `-0.005921`
- `lag_07__T5__flash_duration`: contribution `-0.003576`
- `lag_03__CT_place_ENTRANCE`: contribution `-0.003384`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `-0.003576`
- `lag_09__CT4__flash_duration`: contribution `-0.002850`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.002085`
