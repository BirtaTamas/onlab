# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `54161`, seconds `48.50`, LSTM `0.9030`, delta `+0.1482`
- tick `54097`, seconds `47.50`, LSTM `0.6861`, delta `+0.1283`
- tick `54129`, seconds `48.00`, LSTM `0.7548`, delta `+0.0686`
- tick `54225`, seconds `49.50`, LSTM `0.9557`, delta `+0.0387`
- tick `51121`, seconds `1.00`, LSTM `0.6276`, delta `+0.0312`
- tick `51761`, seconds `11.00`, LSTM `0.6228`, delta `+0.0283`
- tick `54865`, seconds `59.50`, LSTM `0.9733`, delta `+0.0272`
- tick `53873`, seconds `44.00`, LSTM `0.5470`, delta `+0.0229`
- tick `51793`, seconds `11.50`, LSTM `0.6011`, delta `-0.0218`
- tick `51153`, seconds `1.50`, LSTM `0.6083`, delta `-0.0192`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002263`, |coef| `0.002263`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001528`, |coef| `0.001528`
- `lag_00__CT_kills_last_3s`: coefficient `0.001486`, |coef| `0.001486`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001295`, |coef| `0.001295`
- `lag_00__kill_diff_last_3s`: coefficient `0.001239`, |coef| `0.001239`
- `lag_00__damage_diff_last_5s`: coefficient `0.001224`, |coef| `0.001224`
- `lag_00__CT_damage_last_5s`: coefficient `0.001199`, |coef| `0.001199`
- `lag_00__T2__has_bomb`: coefficient `-0.001096`, |coef| `0.001096`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `0.001094`, |coef| `0.001094`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001086`, |coef| `0.001086`
- `lag_02__CT_burning_players`: coefficient `0.000946`, |coef| `0.000946`
- `lag_00__CT3__shots_fired`: coefficient `0.000931`, |coef| `0.000931`
- `lag_03__CT_shots_fired_sum`: coefficient `0.000918`, |coef| `0.000918`
- `lag_00__T2__alive`: coefficient `-0.000842`, |coef| `0.000842`

## Top 10 utility ridge features

- `lag_03__T_B_site_active_infernos`: coefficient `0.000735` (raises CT win probability)
- `lag_07__T3__molly`: coefficient `-0.000691` (lowers CT win probability)
- `lag_13__CT1__smoke`: coefficient `-0.000688` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000602` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000587` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000535` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `0.000525` (raises CT win probability)
- `lag_15__CT1__smoke`: coefficient `-0.000504` (lowers CT win probability)
- `lag_09__T3__molly`: coefficient `-0.000501` (lowers CT win probability)
- `lag_08__T3__molly`: coefficient `-0.000497` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002263` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001637` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001528` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001486` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001295` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001239` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001224` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001199` (raises CT win probability)
- `lag_00__T2__has_bomb`: coefficient `-0.001096` (lowers CT win probability)
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `0.001094` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `54161`, seconds `48.50`, LSTM delta `+0.1482`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015722`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010618`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008996`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.007991`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.005302`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `54097`, seconds `47.50`, LSTM delta `+0.1283`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015722`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.007991`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `+0.004405`
- `lag_00__CT_kills_last_3s`: contribution `+0.004290`
- `lag_00__T2__has_bomb`: contribution `+0.003421`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.002079`

### tick `54129`, seconds `48.00`, LSTM delta `+0.0686`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015722`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010618`
- `lag_01__T_place_SIDEENTRANCE`: contribution `+0.003123`
- `lag_02__CT_place_SIDEENTRANCE`: contribution `+0.002962`
- `lag_00__CT3__shots_fired`: contribution `+0.002393`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `+0.001353`

### tick `54225`, seconds `49.50`, LSTM delta `+0.0387`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.029871`
- `lag_01__CT_shots_fired_sum`: contribution `-0.010618`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008996`
- `lag_00__CT4__shots_fired`: contribution `-0.008181`
- `lag_03__CT_shots_fired_sum`: contribution `+0.006381`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51121`, seconds `1.00`, LSTM delta `+0.0312`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `+0.008667`
- `lag_02__CT_place_SIDEENTRANCE`: contribution `+0.002819`
- `lag_02__CT_place_TSIDELOWER`: contribution `+0.002404`
- `lag_02__T2__has_bomb`: contribution `-0.001521`
- `lag_01__T_velocity_mean`: contribution `+0.001458`

Top utility-only movements:
- `lag_02__CT_molly_inv`: contribution `+0.000583`
- `lag_00__T2__molly`: contribution `+0.000535`
- `lag_00__T3__smoke`: contribution `+0.000521`
- `lag_02__CT_utility_inv`: contribution `+0.000467`
