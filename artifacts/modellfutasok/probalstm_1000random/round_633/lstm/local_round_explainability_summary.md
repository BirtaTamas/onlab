# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `18850`, seconds `77.50`, LSTM `0.1383`, delta `-0.2380`
- tick `18818`, seconds `77.00`, LSTM `0.3762`, delta `+0.1830`
- tick `16610`, seconds `42.50`, LSTM `0.2829`, delta `-0.1154`
- tick `16578`, seconds `42.00`, LSTM `0.3983`, delta `-0.1123`
- tick `16642`, seconds `43.00`, LSTM `0.1901`, delta `-0.0928`
- tick `18882`, seconds `78.00`, LSTM `0.0763`, delta `-0.0619`
- tick `18914`, seconds `78.50`, LSTM `0.0150`, delta `-0.0613`
- tick `16770`, seconds `45.00`, LSTM `0.2649`, delta `+0.0564`
- tick `16898`, seconds `47.00`, LSTM `0.3318`, delta `+0.0530`
- tick `18434`, seconds `71.00`, LSTM `0.2249`, delta `-0.0426`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002407`, |coef| `0.002407`
- `lag_13__CT_place_HOLE`: coefficient `0.002218`, |coef| `0.002218`
- `lag_00__kill_diff_last_3s`: coefficient `0.002046`, |coef| `0.002046`
- `lag_00__T_damage_last_5s`: coefficient `-0.001884`, |coef| `0.001884`
- `lag_00__damage_diff_last_5s`: coefficient `0.001814`, |coef| `0.001814`
- `lag_00__CT_place_HOLE`: coefficient `0.001796`, |coef| `0.001796`
- `lag_01__T_kills_last_3s`: coefficient `-0.001707`, |coef| `0.001707`
- `lag_12__CT_place_HOLE`: coefficient `-0.001703`, |coef| `0.001703`
- `lag_02__CT5__flash_duration`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001596`, |coef| `0.001596`
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001533`, |coef| `0.001533`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001523`, |coef| `0.001523`
- `lag_09__T2__is_walking`: coefficient `0.001503`, |coef| `0.001503`
- `lag_01__T_place_OUTSIDETUNNEL`: coefficient `0.001501`, |coef| `0.001501`
- `lag_15__T_place_OUTSIDETUNNEL`: coefficient `-0.001471`, |coef| `0.001471`

## Top 10 utility ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.001611` (lowers CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.001047` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.000894` (lowers CT win probability)
- `lag_06__T5__molly`: coefficient `0.000869` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.000831` (lowers CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000797` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000757` (lowers CT win probability)
- `lag_15__CT3__smoke`: coefficient `0.000673` (raises CT win probability)
- `lag_10__CT_flash_alpha_mean`: coefficient `0.000638` (raises CT win probability)
- `lag_08__CT_flash_alpha_mean`: coefficient `0.000637` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002407` (lowers CT win probability)
- `lag_13__CT_place_HOLE`: coefficient `0.002218` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002046` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001884` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001814` (raises CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001796` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001707` (lowers CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `-0.001703` (lowers CT win probability)
- `lag_01__CT_place_LONGDOORS`: coefficient `0.001596` (raises CT win probability)
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001533` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18850`, seconds `77.50`, LSTM delta `-0.2380`

Top all feature movements:
- `lag_13__CT_place_HOLE`: contribution `-0.024763`
- `lag_02__CT5__flash_duration`: contribution `-0.008045`
- `lag_00__T_kills_last_3s`: contribution `-0.007626`
- `lag_02__CT_flashed_players`: contribution `-0.005651`
- `lag_00__kill_diff_last_3s`: contribution `-0.004925`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.008045`
- `lag_02__T_B_site_active_infernos`: contribution `-0.002961`
- `lag_02__CT_flash_duration_sum`: contribution `-0.002602`

### tick `18818`, seconds `77.00`, LSTM delta `+0.1830`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `+0.019014`
- `lag_15__CT_place_HOLE`: contribution `+0.013815`
- `lag_03__CT_place_ARAMP`: contribution `+0.006027`
- `lag_00__kill_diff_last_3s`: contribution `+0.004925`
- `lag_00__damage_diff_last_5s`: contribution `+0.003969`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `+0.003116`

### tick `16610`, seconds `42.50`, LSTM delta `-0.1154`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.009489`
- `lag_15__CT_place_ARAMP`: contribution `-0.008651`
- `lag_01__T_place_OUTSIDETUNNEL`: contribution `-0.007501`
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `-0.007353`
- `lag_01__CT_place_LONGDOORS`: contribution `-0.006991`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16578`, seconds `42.00`, LSTM delta `-0.1123`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007626`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.006712`
- `lag_14__CT_place_ARAMP`: contribution `-0.006111`
- `lag_03__CT_place_ARAMP`: contribution `-0.006027`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `-0.005294`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16642`, seconds `43.00`, LSTM delta `-0.0928`

Top all feature movements:
- `lag_12__T_place_TUNNELSTAIRS`: contribution `-0.007095`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `-0.006955`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `-0.006439`
- `lag_02__CT_place_LONGDOORS`: contribution `-0.006254`
- `lag_03__CT_place_ARAMP`: contribution `+0.006027`

Top utility-only movements:
- No utility movement among the top local contributors.
