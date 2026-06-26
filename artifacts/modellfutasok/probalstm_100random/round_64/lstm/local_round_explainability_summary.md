# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `115332`, seconds `55.00`, LSTM `0.6538`, delta `+0.4196`
- tick `116548`, seconds `74.00`, LSTM `0.0687`, delta `-0.3984`
- tick `115396`, seconds `56.00`, LSTM `0.3628`, delta `-0.3350`
- tick `115684`, seconds `60.50`, LSTM `0.4727`, delta `+0.2948`
- tick `115556`, seconds `58.50`, LSTM `0.1168`, delta `-0.2342`
- tick `112260`, seconds `7.00`, LSTM `0.1456`, delta `-0.1720`
- tick `115652`, seconds `60.00`, LSTM `0.1779`, delta `+0.0852`
- tick `115748`, seconds `61.50`, LSTM `0.4978`, delta `+0.0596`
- tick `115812`, seconds `62.50`, LSTM `0.4159`, delta `-0.0533`
- tick `115268`, seconds `54.00`, LSTM `0.2725`, delta `+0.0517`

## Top 15 local ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.005580`, |coef| `0.005580`
- `lag_00__kill_diff_last_3s`: coefficient `0.004240`, |coef| `0.004240`
- `lag_02__CT_place_ARAMP`: coefficient `0.003971`, |coef| `0.003971`
- `lag_00__T_kills_last_3s`: coefficient `-0.003503`, |coef| `0.003503`
- `lag_04__CT_place_SHORTSTAIRS`: coefficient `-0.003466`, |coef| `0.003466`
- `lag_06__CT_place_ARAMP`: coefficient `-0.003412`, |coef| `0.003412`
- `lag_00__damage_diff_last_5s`: coefficient `0.003317`, |coef| `0.003317`
- `lag_13__T_utility_damage_last_5s`: coefficient `0.002749`, |coef| `0.002749`
- `lag_12__T_utility_damage_last_5s`: coefficient `0.002381`, |coef| `0.002381`
- `lag_13__T1__duck_amount`: coefficient `-0.002266`, |coef| `0.002266`
- `lag_06__CT_place_EXTENDEDA`: coefficient `0.002250`, |coef| `0.002250`
- `lag_00__CT_spread_xy`: coefficient `0.002229`, |coef| `0.002229`
- `lag_00__T_damage_last_5s`: coefficient `-0.002220`, |coef| `0.002220`
- `lag_12__T_place_MIDDOORS`: coefficient `0.002115`, |coef| `0.002115`
- `lag_14__T_utility_damage_last_5s`: coefficient `0.002015`, |coef| `0.002015`

## Top 10 utility ridge features

- `lag_13__T_utility_damage_last_5s`: coefficient `0.002749` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.002381` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.002015` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `0.001904` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `0.001857` (raises CT win probability)
- `lag_11__T_utility_damage_last_5s`: coefficient `0.001723` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `-0.001517` (lowers CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001415` (lowers CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001412` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001392` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.005580` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004240` (raises CT win probability)
- `lag_02__CT_place_ARAMP`: coefficient `0.003971` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003503` (lowers CT win probability)
- `lag_04__CT_place_SHORTSTAIRS`: coefficient `-0.003466` (lowers CT win probability)
- `lag_06__CT_place_ARAMP`: coefficient `-0.003412` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003317` (raises CT win probability)
- `lag_13__T1__duck_amount`: coefficient `-0.002266` (lowers CT win probability)
- `lag_06__CT_place_EXTENDEDA`: coefficient `0.002250` (raises CT win probability)
- `lag_00__CT_spread_xy`: coefficient `0.002229` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `115332`, seconds `55.00`, LSTM delta `+0.4196`

Top all feature movements:
- `lag_04__CT_place_SHORTSTAIRS`: contribution `+0.019319`
- `lag_06__CT_place_EXTENDEDA`: contribution `+0.012630`
- `lag_00__kill_diff_last_3s`: contribution `+0.010205`
- `lag_12__T_place_MIDDOORS`: contribution `+0.008990`
- `lag_13__T1__duck_amount`: contribution `+0.008873`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116548`, seconds `74.00`, LSTM delta `-0.3984`

Top all feature movements:
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.042793`
- `lag_02__CT_place_ARAMP`: contribution `-0.024737`
- `lag_06__CT_place_ARAMP`: contribution `-0.021254`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.011277`
- `lag_00__T_kills_last_3s`: contribution `-0.011099`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `-0.009420`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.005440`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.004604`

### tick `115396`, seconds `56.00`, LSTM delta `-0.3350`

Top all feature movements:
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.019319`
- `lag_00__T_kills_last_3s`: contribution `-0.011099`
- `lag_00__kill_diff_last_3s`: contribution `-0.010205`
- `lag_04__CT_shots_fired_sum`: contribution `-0.009263`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.009035`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115684`, seconds `60.50`, LSTM delta `+0.2948`

Top all feature movements:
- `lag_00__CT_place_UPPERTUNNEL`: contribution `+0.042793`
- `lag_04__CT_place_SHORTSTAIRS`: contribution `+0.019319`
- `lag_00__kill_diff_last_3s`: contribution `+0.010205`
- `lag_13__CT_place_EXTENDEDA`: contribution `+0.008978`
- `lag_13__T1__duck_amount`: contribution `+0.008873`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115556`, seconds `58.50`, LSTM delta `-0.2342`

Top all feature movements:
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.019319`
- `lag_00__T_kills_last_3s`: contribution `-0.011099`
- `lag_00__kill_diff_last_3s`: contribution `-0.010205`
- `lag_12__T_place_MIDDOORS`: contribution `-0.008990`
- `lag_13__CT_place_EXTENDEDA`: contribution `-0.008978`

Top utility-only movements:
- No utility movement among the top local contributors.
