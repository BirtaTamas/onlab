# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `18718`, seconds `24.50`, LSTM `0.8121`, delta `+0.2120`
- tick `23838`, seconds `104.50`, LSTM `0.6836`, delta `-0.1863`
- tick `18462`, seconds `20.50`, LSTM `0.6519`, delta `-0.1808`
- tick `18334`, seconds `18.50`, LSTM `0.8285`, delta `+0.1639`
- tick `18174`, seconds `16.00`, LSTM `0.6642`, delta `+0.1597`
- tick `24446`, seconds `114.00`, LSTM `0.9166`, delta `+0.1331`
- tick `23870`, seconds `105.00`, LSTM `0.7796`, delta `+0.0960`
- tick `23902`, seconds `105.50`, LSTM `0.6868`, delta `-0.0928`
- tick `19454`, seconds `36.00`, LSTM `0.9650`, delta `+0.0614`
- tick `24222`, seconds `110.50`, LSTM `0.6995`, delta `+0.0609`

## Top 15 local ridge features

- `lag_09__T_bomb_zone_count`: coefficient `-0.003752`, |coef| `0.003752`
- `lag_00__kill_diff_last_3s`: coefficient `0.003441`, |coef| `0.003441`
- `lag_00__CT_defusing_count`: coefficient `0.003017`, |coef| `0.003017`
- `lag_08__T_velocity_mean`: coefficient `-0.002575`, |coef| `0.002575`
- `lag_00__damage_diff_last_5s`: coefficient `0.002326`, |coef| `0.002326`
- `lag_00__CT_kills_last_3s`: coefficient `0.002304`, |coef| `0.002304`
- `lag_09__T_velocity_mean`: coefficient `0.002276`, |coef| `0.002276`
- `lag_12__T5__flash_duration`: coefficient `-0.002237`, |coef| `0.002237`
- `lag_07__T_velocity_mean`: coefficient `0.002154`, |coef| `0.002154`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002126`, |coef| `0.002126`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002063`, |coef| `0.002063`
- `lag_00__T_kills_last_3s`: coefficient `-0.002000`, |coef| `0.002000`
- `lag_08__T_bomb_zone_count`: coefficient `-0.001945`, |coef| `0.001945`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001755`, |coef| `0.001755`
- `lag_07__T3__flash_duration`: coefficient `-0.001716`, |coef| `0.001716`

## Top 10 utility ridge features

- `lag_12__T5__flash_duration`: coefficient `-0.002237` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002126` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001755` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.001716` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.001712` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001679` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001516` (raises CT win probability)
- `lag_13__T3__flash_duration`: coefficient `-0.001490` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.001490` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.001380` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_bomb_zone_count`: coefficient `-0.003752` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003441` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003017` (raises CT win probability)
- `lag_08__T_velocity_mean`: coefficient `-0.002575` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002326` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002304` (raises CT win probability)
- `lag_09__T_velocity_mean`: coefficient `0.002276` (raises CT win probability)
- `lag_07__T_velocity_mean`: coefficient `0.002154` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002063` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002000` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `18718`, seconds `24.50`, LSTM delta `+0.2120`

Top all feature movements:
- `lag_12__T5__flash_duration`: contribution `+0.012155`
- `lag_05__T3__flash_duration`: contribution `+0.009719`
- `lag_00__kill_diff_last_3s`: contribution `+0.008281`
- `lag_00__CT_kills_last_3s`: contribution `+0.006652`
- `lag_00__damage_diff_last_5s`: contribution `+0.005247`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `+0.012155`
- `lag_05__T3__flash_duration`: contribution `+0.009719`
- `lag_12__T_flash_duration_sum`: contribution `+0.003779`

### tick `23838`, seconds `104.50`, LSTM delta `-0.1863`

Top all feature movements:
- `lag_09__T_bomb_zone_count`: contribution `-0.021840`
- `lag_08__T_velocity_mean`: contribution `-0.020065`
- `lag_07__T_velocity_mean`: contribution `-0.016213`
- `lag_08__T3__flash_duration`: contribution `-0.010983`
- `lag_00__kill_diff_last_3s`: contribution `-0.008281`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.010983`
- `lag_09__T_utility_damage_last_5s`: contribution `-0.002821`

### tick `18462`, seconds `20.50`, LSTM delta `-0.1808`

Top all feature movements:
- `lag_12__T5__flash_duration`: contribution `-0.016164`
- `lag_12__T_flash_duration_sum`: contribution `-0.011162`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.009949`
- `lag_00__kill_diff_last_3s`: contribution `-0.008281`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.006819`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.016164`
- `lag_12__T_flash_duration_sum`: contribution `-0.011162`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.009949`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.006819`
- `lag_12__T3__flash_duration`: contribution `-0.006601`

### tick `18334`, seconds `18.50`, LSTM delta `+0.1639`

Top all feature movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.012395`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.009634`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.008715`
- `lag_00__kill_diff_last_3s`: contribution `+0.008281`
- `lag_08__T3__flash_duration`: contribution `-0.008003`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.012395`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.009634`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.008715`
- `lag_08__T3__flash_duration`: contribution `-0.008003`
- `lag_08__T5__flash_duration`: contribution `+0.006040`

### tick `18174`, seconds `16.00`, LSTM delta `+0.1597`

Top all feature movements:
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.014096`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.013666`
- `lag_00__kill_diff_last_3s`: contribution `+0.008281`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.006838`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.006812`

Top utility-only movements:
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.014096`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.013666`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.006838`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.006812`
- `lag_03__T5__flash_duration`: contribution `+0.006059`
