# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `86860`, seconds `56.00`, LSTM `0.8988`, delta `+0.1194`
- tick `88876`, seconds `87.50`, LSTM `0.9425`, delta `+0.1157`
- tick `84332`, seconds `16.50`, LSTM `0.7417`, delta `+0.1063`
- tick `87564`, seconds `67.00`, LSTM `0.8667`, delta `-0.0725`
- tick `87116`, seconds `60.00`, LSTM `0.9586`, delta `+0.0640`
- tick `87756`, seconds `70.00`, LSTM `0.8264`, delta `+0.0614`
- tick `87660`, seconds `68.50`, LSTM `0.7655`, delta `-0.0439`
- tick `84012`, seconds `11.50`, LSTM `0.6566`, delta `-0.0411`
- tick `84652`, seconds `21.50`, LSTM `0.7022`, delta `-0.0376`
- tick `85036`, seconds `27.50`, LSTM `0.7304`, delta `-0.0337`

## Top 15 local ridge features

- `lag_01__CT_place_VENDING`: coefficient `0.002110`, |coef| `0.002110`
- `lag_01__T_place_MINI`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_00__CT_kills_last_3s`: coefficient `0.001292`, |coef| `0.001292`
- `lag_01__CT_place_SECRET`: coefficient `-0.001264`, |coef| `0.001264`
- `lag_00__kill_diff_last_3s`: coefficient `0.001224`, |coef| `0.001224`
- `lag_00__CT_damage_last_5s`: coefficient `0.001108`, |coef| `0.001108`
- `lag_03__T_place_MINI`: coefficient `0.001106`, |coef| `0.001106`
- `lag_00__CT_place_VENDING`: coefficient `0.001034`, |coef| `0.001034`
- `lag_00__damage_diff_last_5s`: coefficient `0.001025`, |coef| `0.001025`
- `lag_09__CT_place_VENDING`: coefficient `0.000953`, |coef| `0.000953`
- `lag_00__T_place_SILO`: coefficient `-0.000944`, |coef| `0.000944`
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000924`, |coef| `0.000924`
- `lag_01__CT_place_LOBBY`: coefficient `-0.000904`, |coef| `0.000904`
- `lag_06__CT_place_VENDING`: coefficient `0.000882`, |coef| `0.000882`
- `lag_12__CT5__flash_duration`: coefficient `-0.000843`, |coef| `0.000843`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `0.000924` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000843` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000687` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000653` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000614` (lowers CT win probability)
- `lag_06__CT5__molly`: coefficient `-0.000586` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `-0.000584` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.000528` (lowers CT win probability)
- `lag_14__CT_B_site_active_infernos`: coefficient `0.000486` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000419` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_VENDING`: coefficient `0.002110` (raises CT win probability)
- `lag_01__T_place_MINI`: coefficient `-0.001378` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001292` (raises CT win probability)
- `lag_01__CT_place_SECRET`: coefficient `-0.001264` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001224` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001108` (raises CT win probability)
- `lag_03__T_place_MINI`: coefficient `0.001106` (raises CT win probability)
- `lag_00__CT_place_VENDING`: coefficient `0.001034` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001025` (raises CT win probability)
- `lag_09__CT_place_VENDING`: coefficient `0.000953` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `86860`, seconds `56.00`, LSTM delta `+0.1194`

Top all feature movements:
- `lag_01__CT_place_VENDING`: contribution `+0.036160`
- `lag_01__CT_place_LOBBY`: contribution `+0.007403`
- `lag_00__T_place_TROPHY`: contribution `+0.004731`
- `lag_00__CT_kills_last_3s`: contribution `+0.003731`
- `lag_00__kill_diff_last_3s`: contribution `+0.002946`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.002359`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.002304`
- `lag_06__CT5__molly`: contribution `+0.001454`

### tick `88876`, seconds `87.50`, LSTM delta `+0.1157`

Top all feature movements:
- `lag_01__T_place_MINI`: contribution `+0.019166`
- `lag_03__T_place_MINI`: contribution `+0.015383`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.006596`
- `lag_12__CT5__flash_duration`: contribution `+0.006004`
- `lag_15__CT_place_MINI`: contribution `+0.004457`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.006596`
- `lag_12__CT5__flash_duration`: contribution `+0.006004`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002638`

### tick `84332`, seconds `16.50`, LSTM delta `+0.1063`

Top all feature movements:
- `lag_00__T_place_SILO`: contribution `+0.006415`
- `lag_07__CT_place_HUTROOF`: contribution `+0.004660`
- `lag_07__T_place_SQUEAKY`: contribution `+0.004227`
- `lag_11__CT_place_HUTROOF`: contribution `+0.003958`
- `lag_00__CT_kills_last_3s`: contribution `+0.003731`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001865`

### tick `87564`, seconds `67.00`, LSTM delta `-0.0725`

Top all feature movements:
- `lag_01__CT_place_SECRET`: contribution `-0.013007`
- `lag_12__T_place_TROPHY`: contribution `-0.003590`
- `lag_12__T_place_VENDING`: contribution `-0.003413`
- `lag_14__CT_place_LOBBY`: contribution `-0.003052`
- `lag_00__kill_diff_last_3s`: contribution `-0.002946`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `87116`, seconds `60.00`, LSTM delta `+0.0640`

Top all feature movements:
- `lag_09__CT_place_VENDING`: contribution `+0.016330`
- `lag_05__CT_place_VENDING`: contribution `-0.010770`
- `lag_00__CT_damage_last_5s`: contribution `+0.003744`
- `lag_00__CT_kills_last_3s`: contribution `+0.003731`
- `lag_00__CT_place_LOBBY`: contribution `+0.003637`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001865`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.001670`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.001366`
