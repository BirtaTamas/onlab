# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `137835`, seconds `26.50`, LSTM `0.1783`, delta `-0.2877`
- tick `137867`, seconds `27.00`, LSTM `0.4231`, delta `+0.2448`
- tick `137419`, seconds `20.00`, LSTM `0.3171`, delta `-0.2269`
- tick `141707`, seconds `87.00`, LSTM `0.1280`, delta `-0.2252`
- tick `138091`, seconds `30.50`, LSTM `0.7148`, delta `+0.2243`
- tick `137771`, seconds `25.50`, LSTM `0.4427`, delta `+0.1614`
- tick `140619`, seconds `70.00`, LSTM `0.5981`, delta `-0.1423`
- tick `138027`, seconds `29.50`, LSTM `0.4923`, delta `+0.0651`
- tick `137579`, seconds `22.50`, LSTM `0.2130`, delta `-0.0609`
- tick `140651`, seconds `70.50`, LSTM `0.5385`, delta `-0.0596`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005511`, |coef| `0.005511`
- `lag_00__kill_diff_last_3s`: coefficient `0.005450`, |coef| `0.005450`
- `lag_00__damage_diff_last_5s`: coefficient `0.003897`, |coef| `0.003897`
- `lag_00__T_damage_last_5s`: coefficient `-0.003560`, |coef| `0.003560`
- `lag_12__CT_place_ARCH`: coefficient `0.003548`, |coef| `0.003548`
- `lag_06__T_B_site_active_infernos`: coefficient `-0.003302`, |coef| `0.003302`
- `lag_00__CT_burning_players`: coefficient `0.002795`, |coef| `0.002795`
- `lag_00__CT1__molly`: coefficient `0.002709`, |coef| `0.002709`
- `lag_00__CT1__alive`: coefficient `0.002659`, |coef| `0.002659`
- `lag_00__CT2__alive`: coefficient `0.002627`, |coef| `0.002627`
- `lag_00__CT_place_RUINS`: coefficient `0.002619`, |coef| `0.002619`
- `lag_09__T4__duck_amount`: coefficient `-0.002495`, |coef| `0.002495`
- `lag_13__CT_place_SECONDMID`: coefficient `0.002488`, |coef| `0.002488`
- `lag_00__CT1__armor`: coefficient `0.002461`, |coef| `0.002461`
- `lag_08__T4__flash_duration`: coefficient `0.002460`, |coef| `0.002460`

## Top 10 utility ridge features

- `lag_06__T_B_site_active_infernos`: coefficient `-0.003302` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.002709` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.002460` (raises CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.002424` (lowers CT win probability)
- `lag_10__T1__molly`: coefficient `0.002281` (raises CT win probability)
- `lag_08__T4__smoke`: coefficient `-0.002148` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.001900` (lowers CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.001868` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.001638` (raises CT win probability)
- `lag_06__active_infernos_total`: coefficient `-0.001423` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005511` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005450` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003897` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003560` (lowers CT win probability)
- `lag_12__CT_place_ARCH`: coefficient `0.003548` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.002795` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.002659` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.002627` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.002619` (raises CT win probability)
- `lag_09__T4__duck_amount`: coefficient `-0.002495` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `137835`, seconds `26.50`, LSTM delta `-0.2877`

Top all feature movements:
- `lag_13__CT_place_SECONDMID`: contribution `-0.051008`
- `lag_08__CT_place_BACKALLEY`: contribution `-0.036128`
- `lag_00__T_kills_last_3s`: contribution `-0.017459`
- `lag_03__T_shots_fired_sum`: contribution `-0.014043`
- `lag_00__kill_diff_last_3s`: contribution `-0.013118`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `-0.005768`
- `lag_11__T_utility_damage_last_5s`: contribution `-0.004719`

### tick `137867`, seconds `27.00`, LSTM delta `+0.2448`

Top all feature movements:
- `lag_14__CT_place_SECONDMID`: contribution `+0.050037`
- `lag_09__CT_place_BACKALLEY`: contribution `+0.027081`
- `lag_04__T_shots_fired_sum`: contribution `+0.016095`
- `lag_00__kill_diff_last_3s`: contribution `+0.013118`
- `lag_04__T2__shots_fired`: contribution `+0.010331`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `+0.010036`
- `lag_13__T4__flash_duration`: contribution `+0.009895`
- `lag_06__T_B_site_active_infernos`: contribution `+0.009335`
- `lag_11__T2__flash_duration`: contribution `+0.007414`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.006218`

### tick `137419`, seconds `20.00`, LSTM delta `-0.2269`

Top all feature movements:
- `lag_12__CT_place_BACKALLEY`: contribution `-0.033629`
- `lag_12__CT_place_SECONDMID`: contribution `-0.028916`
- `lag_00__T_kills_last_3s`: contribution `-0.017459`
- `lag_00__kill_diff_last_3s`: contribution `-0.013118`
- `lag_08__T4__flash_duration`: contribution `+0.012813`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.012813`
- `lag_06__T_B_site_active_infernos`: contribution `-0.009335`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.005597`
- `lag_06__T_active_infernos`: contribution `-0.005049`
- `lag_08__T2__flash_duration`: contribution `-0.003958`

### tick `141707`, seconds `87.00`, LSTM delta `-0.2252`

Top all feature movements:
- `lag_08__T4__flash_duration`: contribution `-0.017500`
- `lag_00__T_kills_last_3s`: contribution `-0.017459`
- `lag_12__CT_place_ARCH`: contribution `-0.014477`
- `lag_00__kill_diff_last_3s`: contribution `-0.013118`
- `lag_06__T_B_site_active_infernos`: contribution `-0.009335`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.017500`
- `lag_06__T_B_site_active_infernos`: contribution `-0.009335`
- `lag_00__CT1__molly`: contribution `-0.006743`
- `lag_10__T1__molly`: contribution `-0.005050`
- `lag_06__T_active_infernos`: contribution `-0.005049`

### tick `138091`, seconds `30.50`, LSTM delta `+0.2243`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `+0.014010`
- `lag_00__kill_diff_last_3s`: contribution `+0.013118`
- `lag_11__T2__shots_fired`: contribution `+0.012362`
- `lag_11__CT2__duck_amount`: contribution `+0.006800`
- `lag_07__T1__duck_amount`: contribution `+0.005987`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003281`
