# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `37057`, seconds `79.00`, LSTM `0.3960`, delta `-0.1645`
- tick `37025`, seconds `78.50`, LSTM `0.5605`, delta `+0.1499`
- tick `38657`, seconds `104.00`, LSTM `0.0695`, delta `-0.0991`
- tick `39809`, seconds `122.00`, LSTM `0.1094`, delta `+0.0746`
- tick `38529`, seconds `102.00`, LSTM `0.2228`, delta `-0.0690`
- tick `40257`, seconds `129.00`, LSTM `0.2142`, delta `-0.0609`
- tick `35297`, seconds `51.50`, LSTM `0.5675`, delta `+0.0596`
- tick `39841`, seconds `122.50`, LSTM `0.1648`, delta `+0.0554`
- tick `40001`, seconds `125.00`, LSTM `0.1675`, delta `-0.0532`
- tick `36545`, seconds `71.00`, LSTM `0.5296`, delta `+0.0495`

## Top 15 local ridge features

- `lag_00__T_place_UNDERA`: coefficient `-0.001802`, |coef| `0.001802`
- `lag_04__T_flashed_players`: coefficient `0.001535`, |coef| `0.001535`
- `lag_04__T_place_UNDERA`: coefficient `-0.001416`, |coef| `0.001416`
- `lag_01__T_place_UNDERA`: coefficient `-0.001412`, |coef| `0.001412`
- `lag_00__kill_diff_last_3s`: coefficient `0.001291`, |coef| `0.001291`
- `lag_05__T_place_ARAMP`: coefficient `-0.001257`, |coef| `0.001257`
- `lag_08__T_place_UNDERA`: coefficient `-0.001249`, |coef| `0.001249`
- `lag_08__CT_place_HOLE`: coefficient `-0.001193`, |coef| `0.001193`
- `lag_07__CT_place_BDOORS`: coefficient `0.001168`, |coef| `0.001168`
- `lag_02__T_place_UNDERA`: coefficient `-0.001126`, |coef| `0.001126`
- `lag_07__CT_place_MIDDOORS`: coefficient `-0.001116`, |coef| `0.001116`
- `lag_07__CT_place_HOLE`: coefficient `0.001095`, |coef| `0.001095`
- `lag_13__CT4__duck_amount`: coefficient `-0.001081`, |coef| `0.001081`
- `lag_03__CT_place_BDOORS`: coefficient `0.001071`, |coef| `0.001071`
- `lag_03__T_place_UNDERA`: coefficient `-0.001068`, |coef| `0.001068`

## Top 10 utility ridge features

- `lag_05__T3__flash_duration`: coefficient `-0.000869` (lowers CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.000799` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.000781` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.000680` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `0.000664` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `-0.000611` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000607` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000604` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000583` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `0.000568` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_UNDERA`: coefficient `-0.001802` (lowers CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.001535` (raises CT win probability)
- `lag_04__T_place_UNDERA`: coefficient `-0.001416` (lowers CT win probability)
- `lag_01__T_place_UNDERA`: coefficient `-0.001412` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001291` (raises CT win probability)
- `lag_05__T_place_ARAMP`: coefficient `-0.001257` (lowers CT win probability)
- `lag_08__T_place_UNDERA`: coefficient `-0.001249` (lowers CT win probability)
- `lag_08__CT_place_HOLE`: coefficient `-0.001193` (lowers CT win probability)
- `lag_07__CT_place_BDOORS`: coefficient `0.001168` (raises CT win probability)
- `lag_02__T_place_UNDERA`: coefficient `-0.001126` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `37057`, seconds `79.00`, LSTM delta `-0.1645`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `-0.013322`
- `lag_05__T_flashed_players`: contribution `-0.007004`
- `lag_05__T3__flash_duration`: contribution `-0.006557`
- `lag_04__T_flashed_players`: contribution `-0.005924`
- `lag_00__CT_place_ARAMP`: contribution `-0.005696`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.006557`
- `lag_05__T_flash_duration_sum`: contribution `-0.002193`

### tick `37025`, seconds `78.50`, LSTM delta `+0.1499`

Top all feature movements:
- `lag_07__CT_place_HOLE`: contribution `+0.012228`
- `lag_04__T_flashed_players`: contribution `+0.011849`
- `lag_03__CT_place_HOLE`: contribution `+0.010544`
- `lag_04__T3__flash_duration`: contribution `+0.006023`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.005213`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.006023`
- `lag_04__T_flash_duration_sum`: contribution `+0.002759`

### tick `38657`, seconds `104.00`, LSTM delta `-0.0991`

Top all feature movements:
- `lag_04__T_place_UNDERA`: contribution `-0.022133`
- `lag_05__T_place_ARAMP`: contribution `-0.011371`
- `lag_03__T_place_ARAMP`: contribution `-0.005768`
- `lag_13__CT4__duck_amount`: contribution `-0.003971`
- `lag_00__T_kills_last_3s`: contribution `-0.003308`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39809`, seconds `122.00`, LSTM delta `+0.0746`

Top all feature movements:
- `lag_01__T_place_UNDERA`: contribution `+0.022069`
- `lag_08__T_place_UNDERA`: contribution `+0.019514`
- `lag_05__T_place_UNDERA`: contribution `+0.014387`
- `lag_10__T_place_UNDERA`: contribution `-0.008526`
- `lag_08__T1__is_scoped`: contribution `-0.005621`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38529`, seconds `102.00`, LSTM delta `-0.0690`

Top all feature movements:
- `lag_00__T_place_UNDERA`: contribution `-0.028163`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.005213`
- `lag_01__T_place_ARAMP`: contribution `-0.004621`
- `lag_05__T1__is_scoped`: contribution `-0.004121`
- `lag_04__T_place_EXTENDEDA`: contribution `-0.002748`

Top utility-only movements:
- No utility movement among the top local contributors.
