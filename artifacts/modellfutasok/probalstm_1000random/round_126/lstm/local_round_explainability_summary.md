# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `37253`, seconds `94.50`, LSTM `0.9085`, delta `+0.1296`
- tick `35653`, seconds `69.50`, LSTM `0.8004`, delta `+0.1032`
- tick `37477`, seconds `98.00`, LSTM `0.9612`, delta `+0.0434`
- tick `36965`, seconds `90.00`, LSTM `0.7793`, delta `-0.0384`
- tick `31525`, seconds `5.00`, LSTM `0.7315`, delta `+0.0290`
- tick `35621`, seconds `69.00`, LSTM `0.6971`, delta `+0.0270`
- tick `31589`, seconds `6.00`, LSTM `0.7124`, delta `-0.0259`
- tick `36581`, seconds `84.00`, LSTM `0.8165`, delta `+0.0231`
- tick `36997`, seconds `90.50`, LSTM `0.7563`, delta `-0.0231`
- tick `34757`, seconds `55.50`, LSTM `0.7160`, delta `+0.0223`

## Top 15 local ridge features

- `lag_01__T_place_ARCH`: coefficient `0.002400`, |coef| `0.002400`
- `lag_01__T_burning_players`: coefficient `0.001598`, |coef| `0.001598`
- `lag_01__CT1__flash_duration`: coefficient `-0.001551`, |coef| `0.001551`
- `lag_00__CT_kills_last_3s`: coefficient `0.001349`, |coef| `0.001349`
- `lag_14__CT3__duck_amount`: coefficient `-0.001255`, |coef| `0.001255`
- `lag_00__CT_damage_last_5s`: coefficient `0.001209`, |coef| `0.001209`
- `lag_00__damage_diff_last_5s`: coefficient `0.001181`, |coef| `0.001181`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001166`, |coef| `0.001166`
- `lag_00__kill_diff_last_3s`: coefficient `0.001125`, |coef| `0.001125`
- `lag_00__CT1__flash_duration`: coefficient `-0.001112`, |coef| `0.001112`
- `lag_00__CT5__is_walking`: coefficient `-0.001087`, |coef| `0.001087`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000986`, |coef| `0.000986`
- `lag_09__CT1__flash_duration`: coefficient `0.000973`, |coef| `0.000973`
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000942`, |coef| `0.000942`
- `lag_00__T2__duck_amount`: coefficient `-0.000834`, |coef| `0.000834`

## Top 10 utility ridge features

- `lag_01__CT1__flash_duration`: coefficient `-0.001551` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001112` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000986` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.000973` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000942` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000809` (raises CT win probability)
- `lag_06__CT4__molly`: coefficient `-0.000706` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.000560` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000514` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000514` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_ARCH`: coefficient `0.002400` (raises CT win probability)
- `lag_01__T_burning_players`: coefficient `0.001598` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001349` (raises CT win probability)
- `lag_14__CT3__duck_amount`: coefficient `-0.001255` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001209` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001181` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001166` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001125` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.001087` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.000834` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `37253`, seconds `94.50`, LSTM delta `+0.1296`

Top all feature movements:
- `lag_01__T_place_ARCH`: contribution `+0.044657`
- `lag_01__CT1__flash_duration`: contribution `+0.010204`
- `lag_09__CT1__flash_duration`: contribution `+0.006981`
- `lag_09__T_flashed_players`: contribution `+0.004751`
- `lag_06__CT_place_QUAD`: contribution `+0.004392`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.010204`
- `lag_09__CT1__flash_duration`: contribution `+0.006981`
- `lag_04__T5__flash_duration`: contribution `+0.001265`

### tick `35653`, seconds `69.50`, LSTM delta `+0.1032`

Top all feature movements:
- `lag_01__T_burning_players`: contribution `+0.008097`
- `lag_14__CT3__duck_amount`: contribution `+0.004668`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004049`
- `lag_00__CT_kills_last_3s`: contribution `+0.003896`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003237`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003237`
- `lag_06__CT4__molly`: contribution `+0.001740`

### tick `37477`, seconds `98.00`, LSTM delta `+0.0434`

Top all feature movements:
- `lag_08__T_place_ARCH`: contribution `+0.014844`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004049`
- `lag_00__CT_kills_last_3s`: contribution `+0.003896`
- `lag_13__CT_place_QUAD`: contribution `+0.003249`
- `lag_01__CT_place_QUAD`: contribution `+0.003096`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36965`, seconds `90.00`, LSTM delta `-0.0384`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `-0.007980`
- `lag_00__T_flashed_players`: contribution `-0.003105`
- `lag_00__CT5__is_walking`: contribution `-0.002606`
- `lag_00__CT_walking_count`: contribution `-0.001878`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.001875`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.007980`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.001875`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001657`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.001317`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.001269`

### tick `31525`, seconds `5.00`, LSTM delta `+0.0290`

Top all feature movements:
- `lag_04__CT_place_LIBRARY`: contribution `+0.005144`
- `lag_00__CT_place_LIBRARY`: contribution `+0.004344`
- `lag_05__CT_place_LIBRARY`: contribution `+0.003006`
- `lag_09__T_smokes_last_5s`: contribution `+0.002433`
- `lag_09__T_velocity_mean`: contribution `+0.001358`

Top utility-only movements:
- `lag_09__T_smokes_last_5s`: contribution `+0.002433`
- `lag_10__T4__smoke`: contribution `-0.000300`
- `lag_10__CT4__molly`: contribution `-0.000275`
