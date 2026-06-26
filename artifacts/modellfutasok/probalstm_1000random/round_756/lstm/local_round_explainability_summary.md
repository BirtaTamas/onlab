# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `52432`, seconds `52.50`, LSTM `0.8356`, delta `+0.0725`
- tick `52944`, seconds `60.50`, LSTM `0.9362`, delta `+0.0468`
- tick `52496`, seconds `53.50`, LSTM `0.8912`, delta `+0.0463`
- tick `49136`, seconds `1.00`, LSTM `0.7017`, delta `-0.0378`
- tick `50160`, seconds `17.00`, LSTM `0.7242`, delta `-0.0353`
- tick `51024`, seconds `30.50`, LSTM `0.7591`, delta `+0.0329`
- tick `49264`, seconds `3.00`, LSTM `0.6824`, delta `-0.0258`
- tick `50448`, seconds `21.50`, LSTM `0.7390`, delta `+0.0252`
- tick `49456`, seconds `6.00`, LSTM `0.6976`, delta `+0.0240`
- tick `52400`, seconds `52.00`, LSTM `0.7631`, delta `-0.0236`

## Top 15 local ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.000635`, |coef| `0.000635`
- `lag_00__T_smokes_last_5s`: coefficient `-0.000615`, |coef| `0.000615`
- `lag_04__CT_place_RUINS`: coefficient `0.000501`, |coef| `0.000501`
- `lag_07__CT_place_BALCONY`: coefficient `0.000472`, |coef| `0.000472`
- `lag_06__CT4__flash_duration`: coefficient `0.000436`, |coef| `0.000436`
- `lag_06__T2__flash_duration`: coefficient `0.000428`, |coef| `0.000428`
- `lag_00__CT_kills_last_3s`: coefficient `0.000420`, |coef| `0.000420`
- `lag_01__T_mollies_last_5s`: coefficient `-0.000399`, |coef| `0.000399`
- `lag_00__T_burning_players`: coefficient `-0.000394`, |coef| `0.000394`
- `lag_06__T_flashed_players`: coefficient `0.000389`, |coef| `0.000389`
- `lag_12__CT_place_RUINS`: coefficient `0.000388`, |coef| `0.000388`
- `lag_00__CT_place_ARCH`: coefficient `-0.000383`, |coef| `0.000383`
- `lag_08__T_flashed_players`: coefficient `0.000383`, |coef| `0.000383`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000382`, |coef| `0.000382`
- `lag_01__CT_flashes_last_5s`: coefficient `0.000377`, |coef| `0.000377`

## Top 10 utility ridge features

- `lag_00__T_smokes_last_5s`: coefficient `-0.000615` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000436` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000428` (raises CT win probability)
- `lag_01__T_mollies_last_5s`: coefficient `-0.000399` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000382` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000377` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000375` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000371` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000358` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.000347` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.000635` (lowers CT win probability)
- `lag_04__CT_place_RUINS`: coefficient `0.000501` (raises CT win probability)
- `lag_07__CT_place_BALCONY`: coefficient `0.000472` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000420` (raises CT win probability)
- `lag_00__T_burning_players`: coefficient `-0.000394` (lowers CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.000389` (raises CT win probability)
- `lag_12__CT_place_RUINS`: coefficient `0.000388` (raises CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `-0.000383` (lowers CT win probability)
- `lag_08__T_flashed_players`: coefficient `0.000383` (raises CT win probability)
- `lag_06__CT_place_RUINS`: coefficient `0.000367` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `52432`, seconds `52.50`, LSTM delta `+0.0725`

Top all feature movements:
- `lag_06__T_flashed_players`: contribution `+0.003757`
- `lag_06__T2__flash_duration`: contribution `+0.003123`
- `lag_08__T_flashes_last_5s`: contribution `+0.002958`
- `lag_03__T5__flash_duration`: contribution `+0.002866`
- `lag_06__CT4__flash_duration`: contribution `+0.002550`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `+0.003123`
- `lag_08__T_flashes_last_5s`: contribution `+0.002958`
- `lag_03__T5__flash_duration`: contribution `+0.002866`
- `lag_06__CT4__flash_duration`: contribution `+0.002550`
- `lag_06__CT3__flash_duration`: contribution `+0.002526`

### tick `52944`, seconds `60.50`, LSTM delta `+0.0468`

Top all feature movements:
- `lag_02__T4__flash_duration`: contribution `+0.001784`
- `lag_06__T_flashed_players`: contribution `-0.001503`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001230`
- `lag_00__CT_kills_last_3s`: contribution `+0.001211`
- `lag_11__CT_place_ARCH`: contribution `+0.001093`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.001784`
- `lag_06__T3__flash_duration`: contribution `+0.001040`
- `lag_09__CT3__flash_duration`: contribution `+0.001008`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.000863`
- `lag_06__T_flash_duration_sum`: contribution `-0.000761`

### tick `52496`, seconds `53.50`, LSTM delta `+0.0463`

Top all feature movements:
- `lag_08__T_flashed_players`: contribution `+0.003693`
- `lag_08__T2__flash_duration`: contribution `+0.002129`
- `lag_08__CT3__flash_duration`: contribution `+0.001383`
- `lag_06__CT_place_RUINS`: contribution `+0.001281`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.001258`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `+0.002129`
- `lag_08__CT3__flash_duration`: contribution `+0.001383`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.001258`
- `lag_08__T_flash_duration_sum`: contribution `+0.001206`
- `lag_10__T_flashes_last_5s`: contribution `+0.001154`

### tick `49136`, seconds `1.00`, LSTM delta `-0.0378`

Top all feature movements:
- `lag_00__T_smokes_last_5s`: contribution `-0.009017`
- `lag_01__T_mollies_last_5s`: contribution `-0.008202`
- `lag_02__T_smokes_last_5s`: contribution `-0.003236`
- `lag_01__T_he_last_5s`: contribution `-0.003185`
- `lag_01__T_flashes_last_5s`: contribution `-0.002398`

Top utility-only movements:
- `lag_00__T_smokes_last_5s`: contribution `-0.009017`
- `lag_01__T_mollies_last_5s`: contribution `-0.008202`
- `lag_02__T_smokes_last_5s`: contribution `-0.003236`
- `lag_01__T_he_last_5s`: contribution `-0.003185`
- `lag_01__T_flashes_last_5s`: contribution `-0.002398`

### tick `50160`, seconds `17.00`, LSTM delta `-0.0353`

Top all feature movements:
- `lag_09__T_place_KITCHEN`: contribution `-0.009400`
- `lag_06__T_place_DECK`: contribution `-0.007184`
- `lag_06__T_place_KITCHEN`: contribution `-0.006093`
- `lag_00__T_place_DECK`: contribution `-0.004915`
- `lag_14__T_place_UPSTAIRS`: contribution `-0.002738`

Top utility-only movements:
- No utility movement among the top local contributors.
