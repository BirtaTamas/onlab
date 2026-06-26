# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `40848`, seconds `45.00`, LSTM `0.1627`, delta `-0.2614`
- tick `40016`, seconds `32.00`, LSTM `0.2409`, delta `-0.1960`
- tick `40720`, seconds `43.00`, LSTM `0.2790`, delta `+0.1676`
- tick `41360`, seconds `53.00`, LSTM `0.0278`, delta `-0.0583`
- tick `40880`, seconds `45.50`, LSTM `0.1073`, delta `-0.0553`
- tick `40816`, seconds `44.50`, LSTM `0.4240`, delta `+0.0550`
- tick `40912`, seconds `46.00`, LSTM `0.0536`, delta `-0.0537`
- tick `40784`, seconds `44.00`, LSTM `0.3690`, delta `+0.0490`
- tick `40048`, seconds `32.50`, LSTM `0.1936`, delta `-0.0472`
- tick `40688`, seconds `42.50`, LSTM `0.1114`, delta `+0.0431`

## Top 15 local ridge features

- `lag_12__CT_place_VENTS`: coefficient `0.002208`, |coef| `0.002208`
- `lag_12__T5__flash_duration`: coefficient `0.002098`, |coef| `0.002098`
- `lag_13__T1__flash_duration`: coefficient `0.001934`, |coef| `0.001934`
- `lag_00__kill_diff_last_3s`: coefficient `0.001904`, |coef| `0.001904`
- `lag_08__CT5__duck_amount`: coefficient `0.001794`, |coef| `0.001794`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001794`, |coef| `0.001794`
- `lag_00__T_kills_last_3s`: coefficient `-0.001722`, |coef| `0.001722`
- `lag_13__CT_place_DECON`: coefficient `-0.001573`, |coef| `0.001573`
- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.001403`, |coef| `0.001403`
- `lag_03__CT3__shots_fired`: coefficient `0.001375`, |coef| `0.001375`
- `lag_00__CT_place_OBSERVATION`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_05__T_place_RAMP`: coefficient `-0.001328`, |coef| `0.001328`
- `lag_13__T_place_CONTROL`: coefficient `-0.001290`, |coef| `0.001290`
- `lag_08__T5__flash_duration`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_09__T1__flash_duration`: coefficient `-0.001265`, |coef| `0.001265`

## Top 10 utility ridge features

- `lag_12__T5__flash_duration`: coefficient `0.002098` (raises CT win probability)
- `lag_13__T1__flash_duration`: coefficient `0.001934` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.001274` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.001265` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `0.001193` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001177` (lowers CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001113` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001071` (raises CT win probability)
- `lag_11__T_utility_damage_last_5s`: coefficient `-0.001011` (lowers CT win probability)
- `lag_08__CT1__smoke`: coefficient `0.000852` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_VENTS`: coefficient `0.002208` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001904` (raises CT win probability)
- `lag_08__CT5__duck_amount`: coefficient `0.001794` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.001794` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001722` (lowers CT win probability)
- `lag_13__CT_place_DECON`: coefficient `-0.001573` (lowers CT win probability)
- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.001403` (lowers CT win probability)
- `lag_03__CT3__shots_fired`: coefficient `0.001375` (raises CT win probability)
- `lag_00__CT_place_OBSERVATION`: coefficient `-0.001329` (lowers CT win probability)
- `lag_05__T_place_RAMP`: coefficient `-0.001328` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40848`, seconds `45.00`, LSTM delta `-0.2614`

Top all feature movements:
- `lag_12__CT_place_VENTS`: contribution `-0.018530`
- `lag_12__T5__flash_duration`: contribution `-0.013823`
- `lag_13__T1__flash_duration`: contribution `-0.011384`
- `lag_03__CT_shots_fired_sum`: contribution `-0.009969`
- `lag_08__CT5__duck_amount`: contribution `-0.006645`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.013823`
- `lag_13__T1__flash_duration`: contribution `-0.011384`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.005108`
- `lag_11__T_utility_damage_last_5s`: contribution `-0.003753`
- `lag_05__T_A_site_active_infernos`: contribution `-0.003502`

### tick `40016`, seconds `32.00`, LSTM delta `-0.1960`

Top all feature movements:
- `lag_09__CT_place_LOCKERROOM`: contribution `-0.017466`
- `lag_03__CT_place_LOCKERROOM`: contribution `-0.012690`
- `lag_01__CT_place_VENTS`: contribution `-0.007471`
- `lag_01__T_place_CONTROL`: contribution `-0.007396`
- `lag_07__T_place_CONTROL`: contribution `-0.006847`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40720`, seconds `43.00`, LSTM delta `+0.1676`

Top all feature movements:
- `lag_13__CT_place_DECON`: contribution `+0.025019`
- `lag_13__T_place_CONTROL`: contribution `+0.009167`
- `lag_08__CT_place_VENTS`: contribution `+0.008679`
- `lag_14__CT_place_ADMIN`: contribution `+0.008410`
- `lag_08__T5__flash_duration`: contribution `+0.008396`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.008396`
- `lag_09__T1__flash_duration`: contribution `+0.007444`

### tick `41360`, seconds `53.00`, LSTM delta `-0.0583`

Top all feature movements:
- `lag_14__CT_place_OBSERVATION`: contribution `-0.007349`
- `lag_00__T_kills_last_3s`: contribution `-0.005457`
- `lag_00__kill_diff_last_3s`: contribution `-0.004584`
- `lag_02__CT_place_OBSERVATION`: contribution `-0.004000`
- `lag_15__CT4__is_walking`: contribution `-0.002524`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.001765`
- `lag_06__T_B_site_active_infernos`: contribution `-0.001578`

### tick `40880`, seconds `45.50`, LSTM delta `-0.0553`

Top all feature movements:
- `lag_00__CT_place_HELL`: contribution `-0.004825`
- `lag_12__CT5__duck_amount`: contribution `+0.004341`
- `lag_13__T5__flash_duration`: contribution `-0.004203`
- `lag_14__T1__flash_duration`: contribution `-0.003835`
- `lag_13__CT_place_VENTS`: contribution `-0.003698`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.004203`
- `lag_14__T1__flash_duration`: contribution `-0.003835`
- `lag_13__T_flash_duration_sum`: contribution `-0.002866`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.001924`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.001793`
