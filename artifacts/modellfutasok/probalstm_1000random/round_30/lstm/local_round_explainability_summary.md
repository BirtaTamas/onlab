# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `32520`, seconds `113.00`, LSTM `0.7014`, delta `+0.4316`
- tick `31528`, seconds `97.50`, LSTM `0.3836`, delta `-0.3343`
- tick `31016`, seconds `89.50`, LSTM `0.8617`, delta `+0.2987`
- tick `32968`, seconds `120.00`, LSTM `0.8880`, delta `+0.2093`
- tick `30984`, seconds `89.00`, LSTM `0.5629`, delta `+0.1864`
- tick `26056`, seconds `12.00`, LSTM `0.1378`, delta `-0.1557`
- tick `26760`, seconds `23.00`, LSTM `0.2926`, delta `+0.1522`
- tick `31080`, seconds `90.50`, LSTM `0.7713`, delta `-0.0949`
- tick `32552`, seconds `113.50`, LSTM `0.6142`, delta `-0.0872`
- tick `27176`, seconds `29.50`, LSTM `0.2770`, delta `+0.0830`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006215`, |coef| `0.006215`
- `lag_00__CT_shots_fired_sum`: coefficient `0.005872`, |coef| `0.005872`
- `lag_00__CT_kills_last_3s`: coefficient `0.005644`, |coef| `0.005644`
- `lag_06__T5__flash_duration`: coefficient `-0.005033`, |coef| `0.005033`
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.004978`, |coef| `0.004978`
- `lag_00__T_shots_fired_sum`: coefficient `-0.004140`, |coef| `0.004140`
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.003899`, |coef| `0.003899`
- `lag_05__T_bomb_zone_count`: coefficient `-0.003740`, |coef| `0.003740`
- `lag_00__CT5__shots_fired`: coefficient `0.003477`, |coef| `0.003477`
- `lag_00__damage_diff_last_5s`: coefficient `0.003303`, |coef| `0.003303`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003254`, |coef| `0.003254`
- `lag_00__CT_defusing_count`: coefficient `0.003144`, |coef| `0.003144`
- `lag_14__CT_shots_fired_sum`: coefficient `0.003096`, |coef| `0.003096`
- `lag_00__T_macro_B`: coefficient `-0.003059`, |coef| `0.003059`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.003059`, |coef| `0.003059`

## Top 10 utility ridge features

- `lag_06__T5__flash_duration`: coefficient `-0.005033` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.004978` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.003899` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002883` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.002620` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.002166` (lowers CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.002146` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002131` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.002126` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.002099` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006215` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.005872` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005644` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.004140` (lowers CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.003740` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.003477` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003303` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003254` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003144` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.003096` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `32520`, seconds `113.00`, LSTM delta `+0.4316`

Top all feature movements:
- `lag_06__T5__flash_duration`: contribution `+0.035156`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.029038`
- `lag_00__CT_shots_fired_sum`: contribution `+0.020399`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.018658`
- `lag_00__T_shots_fired_sum`: contribution `+0.018624`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.035156`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.029038`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.018658`

### tick `31528`, seconds `97.50`, LSTM delta `-0.3343`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.027966`
- `lag_05__T_bomb_zone_count`: contribution `-0.021771`
- `lag_00__T_bomb_zone_count`: contribution `-0.018942`
- `lag_00__T_shots_fired_sum`: contribution `-0.015520`
- `lag_00__kill_diff_last_3s`: contribution `-0.014959`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `-0.010793`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.008271`
- `lag_08__T5__flash_duration`: contribution `-0.007965`
- `lag_14__CT2__flash_duration`: contribution `-0.005361`

### tick `31016`, seconds `89.50`, LSTM delta `+0.2987`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.019031`
- `lag_01__CT_shots_fired_sum`: contribution `+0.016949`
- `lag_00__CT_kills_last_3s`: contribution `+0.016294`
- `lag_00__kill_diff_last_3s`: contribution `+0.014959`
- `lag_07__T_flashed_players`: contribution `+0.012630`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.019031`
- `lag_01__T2__flash_duration`: contribution `+0.012385`
- `lag_04__T_flash_duration_sum`: contribution `+0.011578`
- `lag_04__T4__flash_duration`: contribution `+0.011223`
- `lag_04__T2__flash_duration`: contribution `+0.006916`

### tick `32968`, seconds `120.00`, LSTM delta `+0.2093`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.020399`
- `lag_00__T_flash_alpha_mean`: contribution `+0.017491`
- `lag_00__CT_kills_last_3s`: contribution `+0.016294`
- `lag_00__kill_diff_last_3s`: contribution `+0.014959`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.011980`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017491`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.011980`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.010171`

### tick `30984`, seconds `89.00`, LSTM delta `+0.1864`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.032638`
- `lag_00__CT_kills_last_3s`: contribution `+0.016294`
- `lag_00__kill_diff_last_3s`: contribution `+0.014959`
- `lag_03__T_flash_duration_sum`: contribution `+0.011227`
- `lag_03__T4__flash_duration`: contribution `+0.007240`

Top utility-only movements:
- `lag_03__T_flash_duration_sum`: contribution `+0.011227`
- `lag_03__T4__flash_duration`: contribution `+0.007240`
- `lag_00__T2__flash_duration`: contribution `+0.007052`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.004414`
- `lag_03__T1__flash_duration`: contribution `+0.003999`
