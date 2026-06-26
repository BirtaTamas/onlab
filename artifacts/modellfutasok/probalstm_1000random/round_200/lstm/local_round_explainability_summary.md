# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `110224`, seconds `27.50`, LSTM `0.3451`, delta `-0.1648`
- tick `110256`, seconds `28.00`, LSTM `0.2563`, delta `-0.0889`
- tick `110448`, seconds `31.00`, LSTM `0.0406`, delta `-0.0787`
- tick `110288`, seconds `28.50`, LSTM `0.1790`, delta `-0.0773`
- tick `110320`, seconds `29.00`, LSTM `0.1223`, delta `-0.0567`
- tick `111216`, seconds `43.00`, LSTM `0.0260`, delta `-0.0232`
- tick `111184`, seconds `42.50`, LSTM `0.0492`, delta `-0.0206`
- tick `111152`, seconds `42.00`, LSTM `0.0699`, delta `+0.0173`
- tick `110352`, seconds `29.50`, LSTM `0.1060`, delta `-0.0163`
- tick `110416`, seconds `30.50`, LSTM `0.1193`, delta `+0.0147`

## Top 15 local ridge features

- `lag_00__T_place_CONTROL`: coefficient `-0.002601`, |coef| `0.002601`
- `lag_05__T_place_SILO`: coefficient `0.001916`, |coef| `0.001916`
- `lag_03__T_place_VENDING`: coefficient `0.001794`, |coef| `0.001794`
- `lag_00__T_place_VENDING`: coefficient `0.001618`, |coef| `0.001618`
- `lag_03__T_place_TROPHY`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_01__T_place_CONTROL`: coefficient `-0.001555`, |coef| `0.001555`
- `lag_00__T_kills_last_3s`: coefficient `-0.001292`, |coef| `0.001292`
- `lag_06__T_place_SILO`: coefficient `0.001264`, |coef| `0.001264`
- `lag_01__T_place_VENDING`: coefficient `0.001220`, |coef| `0.001220`
- `lag_04__T_place_VENDING`: coefficient `0.001208`, |coef| `0.001208`
- `lag_04__CT_flashed_players`: coefficient `-0.001205`, |coef| `0.001205`
- `lag_00__CT4__alive`: coefficient `0.001160`, |coef| `0.001160`
- `lag_00__CT4__hp`: coefficient `0.001143`, |coef| `0.001143`
- `lag_00__CT_place_CATWALK`: coefficient `0.001132`, |coef| `0.001132`
- `lag_00__CT_flashed_players`: coefficient `0.001098`, |coef| `0.001098`

## Top 10 utility ridge features

- `lag_04__CT1__flash_duration`: coefficient `-0.000777` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000731` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000639` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.000625` (lowers CT win probability)
- `lag_07__T3__flash`: coefficient `0.000613` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000577` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000557` (raises CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `-0.000485` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.000482` (lowers CT win probability)
- `lag_08__T3__flash`: coefficient `0.000460` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_CONTROL`: coefficient `-0.002601` (lowers CT win probability)
- `lag_05__T_place_SILO`: coefficient `0.001916` (raises CT win probability)
- `lag_03__T_place_VENDING`: coefficient `0.001794` (raises CT win probability)
- `lag_00__T_place_VENDING`: coefficient `0.001618` (raises CT win probability)
- `lag_03__T_place_TROPHY`: coefficient `-0.001568` (lowers CT win probability)
- `lag_01__T_place_CONTROL`: coefficient `-0.001555` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001292` (lowers CT win probability)
- `lag_06__T_place_SILO`: coefficient `0.001264` (raises CT win probability)
- `lag_01__T_place_VENDING`: coefficient `0.001220` (raises CT win probability)
- `lag_04__T_place_VENDING`: coefficient `0.001208` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `110224`, seconds `27.50`, LSTM delta `-0.1648`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `-0.018480`
- `lag_05__T_place_SILO`: contribution `-0.013019`
- `lag_03__T_place_TROPHY`: contribution `-0.009946`
- `lag_03__T_place_VENDING`: contribution `-0.009094`
- `lag_00__T_place_VENDING`: contribution `-0.008203`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110256`, seconds `28.00`, LSTM delta `-0.0889`

Top all feature movements:
- `lag_01__T_place_CONTROL`: contribution `-0.011053`
- `lag_06__T_place_SILO`: contribution `-0.008588`
- `lag_01__T_place_VENDING`: contribution `-0.006185`
- `lag_04__T_place_VENDING`: contribution `-0.006123`
- `lag_04__bomb_events_last_5s`: contribution `-0.003774`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.001554`

### tick `110448`, seconds `31.00`, LSTM delta `-0.0787`

Top all feature movements:
- `lag_07__T_place_CONTROL`: contribution `-0.006621`
- `lag_05__T_place_CONTROL`: contribution `-0.006582`
- `lag_00__T_kills_last_3s`: contribution `-0.004093`
- `lag_12__T_place_SILO`: contribution `-0.004022`
- `lag_10__T_place_TROPHY`: contribution `-0.003911`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110288`, seconds `28.50`, LSTM delta `-0.0773`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `-0.018480`
- `lag_00__T_place_TROPHY`: contribution `-0.006928`
- `lag_02__T_place_CONTROL`: contribution `-0.005825`
- `lag_02__T_place_VENDING`: contribution `-0.005506`
- `lag_07__T_place_SILO`: contribution `-0.005229`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110320`, seconds `29.00`, LSTM delta `-0.0567`

Top all feature movements:
- `lag_01__T_place_CONTROL`: contribution `-0.011053`
- `lag_03__T_place_VENDING`: contribution `-0.009094`
- `lag_00__T_place_TROPHY`: contribution `-0.006928`
- `lag_03__T_place_CONTROL`: contribution `-0.004460`
- `lag_08__T_place_SILO`: contribution `-0.003556`

Top utility-only movements:
- No utility movement among the top local contributors.
