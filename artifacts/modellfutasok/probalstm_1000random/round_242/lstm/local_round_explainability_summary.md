# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `18788`, seconds `11.50`, LSTM `0.1995`, delta `-0.3303`
- tick `22180`, seconds `64.50`, LSTM `0.7735`, delta `+0.2460`
- tick `19876`, seconds `28.50`, LSTM `0.1250`, delta `-0.2441`
- tick `21540`, seconds `54.50`, LSTM `0.3767`, delta `+0.2176`
- tick `19012`, seconds `15.00`, LSTM `0.3637`, delta `+0.1873`
- tick `21924`, seconds `60.50`, LSTM `0.6378`, delta `+0.1413`
- tick `23172`, seconds `80.00`, LSTM `0.8305`, delta `+0.1096`
- tick `19684`, seconds `25.50`, LSTM `0.3638`, delta `-0.0933`
- tick `23492`, seconds `85.00`, LSTM `0.9479`, delta `+0.0862`
- tick `24100`, seconds `94.50`, LSTM `0.9525`, delta `+0.0724`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005653`, |coef| `0.005653`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003892`, |coef| `0.003892`
- `lag_10__CT_flashes_last_5s`: coefficient `0.003311`, |coef| `0.003311`
- `lag_00__kill_diff_last_3s`: coefficient `0.003075`, |coef| `0.003075`
- `lag_13__CT_place_SECONDMID`: coefficient `-0.002488`, |coef| `0.002488`
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.002463`, |coef| `0.002463`
- `lag_15__T_place_TRAMP`: coefficient `0.002342`, |coef| `0.002342`
- `lag_00__damage_diff_last_5s`: coefficient `0.002270`, |coef| `0.002270`
- `lag_09__CT_place_ARCH`: coefficient `-0.002174`, |coef| `0.002174`
- `lag_10__T_place_LOWERMID`: coefficient `0.002140`, |coef| `0.002140`
- `lag_00__CT_kills_last_3s`: coefficient `0.002057`, |coef| `0.002057`
- `lag_14__T_place_LOWERMID`: coefficient `-0.002047`, |coef| `0.002047`
- `lag_12__CT_shots_fired_sum`: coefficient `0.001992`, |coef| `0.001992`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001948`, |coef| `0.001948`
- `lag_13__CT_place_LIBRARY`: coefficient `0.001900`, |coef| `0.001900`

## Top 10 utility ridge features

- `lag_10__CT_flashes_last_5s`: coefficient `0.003311` (raises CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.002463` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001872` (lowers CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `0.001733` (raises CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.001662` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.001511` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001354` (lowers CT win probability)
- `lag_10__CT_active_infernos`: coefficient `0.001335` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001329` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001220` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005653` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003892` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003075` (raises CT win probability)
- `lag_13__CT_place_SECONDMID`: coefficient `-0.002488` (lowers CT win probability)
- `lag_15__T_place_TRAMP`: coefficient `0.002342` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002270` (raises CT win probability)
- `lag_09__CT_place_ARCH`: coefficient `-0.002174` (lowers CT win probability)
- `lag_10__T_place_LOWERMID`: coefficient `0.002140` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002057` (raises CT win probability)
- `lag_14__T_place_LOWERMID`: coefficient `-0.002047` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `18788`, seconds `11.50`, LSTM delta `-0.3303`

Top all feature movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.036408`
- `lag_10__T_place_LOWERMID`: contribution `-0.014235`
- `lag_14__T_place_LOWERMID`: contribution `-0.013619`
- `lag_13__CT_place_LIBRARY`: contribution `-0.012185`
- `lag_08__T_place_LOWERMID`: contribution `-0.010275`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.036408`
- `lag_00__CT1__flash`: contribution `-0.004756`

### tick `22180`, seconds `64.50`, LSTM delta `+0.2460`

Top all feature movements:
- `lag_13__CT_place_SECONDMID`: contribution `+0.051023`
- `lag_00__T_shots_fired_sum`: contribution `+0.014591`
- `lag_14__T_shots_fired_sum`: contribution `+0.007492`
- `lag_00__kill_diff_last_3s`: contribution `+0.007401`
- `lag_11__T1__duck_amount`: contribution `+0.007222`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `+0.005869`
- `lag_09__CT3__flash_duration`: contribution `+0.003829`

### tick `19876`, seconds `28.50`, LSTM delta `-0.2441`

Top all feature movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.015470`
- `lag_00__T_shots_fired_sum`: contribution `-0.014591`
- `lag_09__CT_place_ARCH`: contribution `-0.008872`
- `lag_00__kill_diff_last_3s`: contribution `-0.007401`
- `lag_04__T5__duck_amount`: contribution `-0.007049`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.015470`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.006601`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.005954`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.005331`

### tick `21540`, seconds `54.50`, LSTM delta `+0.2176`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.070037`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013535`
- `lag_02__T_shots_fired_sum`: contribution `+0.010781`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010305`
- `lag_00__kill_diff_last_3s`: contribution `+0.007401`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.003488`
- `lag_00__CT3__flash_duration`: contribution `+0.003408`

### tick `19012`, seconds `15.00`, LSTM delta `+0.1873`

Top all feature movements:
- `lag_15__T_place_TRAMP`: contribution `+0.013708`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009475`
- `lag_00__T_shots_fired_sum`: contribution `+0.008755`
- `lag_00__kill_diff_last_3s`: contribution `+0.007401`
- `lag_00__CT_kills_last_3s`: contribution `+0.005939`

Top utility-only movements:
- `lag_07__CT1__flash`: contribution `+0.002792`
- `lag_08__T_B_site_active_infernos`: contribution `+0.002554`
