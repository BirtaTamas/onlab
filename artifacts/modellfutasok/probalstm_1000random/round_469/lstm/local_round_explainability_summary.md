# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `119479`, seconds `35.50`, LSTM `0.7297`, delta `+0.1665`
- tick `119639`, seconds `38.00`, LSTM `0.8739`, delta `+0.1037`
- tick `119735`, seconds `39.50`, LSTM `0.9447`, delta `+0.0650`
- tick `119511`, seconds `36.00`, LSTM `0.7839`, delta `+0.0542`
- tick `119575`, seconds `37.00`, LSTM `0.7330`, delta `-0.0459`
- tick `119351`, seconds `33.50`, LSTM `0.5607`, delta `+0.0399`
- tick `119607`, seconds `37.50`, LSTM `0.7701`, delta `+0.0371`
- tick `119863`, seconds `41.50`, LSTM `0.9697`, delta `+0.0145`
- tick `119767`, seconds `40.00`, LSTM `0.9568`, delta `+0.0121`
- tick `119383`, seconds `34.00`, LSTM `0.5703`, delta `+0.0096`

## Top 15 local ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.003161`, |coef| `0.003161`
- `lag_00__CT_flashed_players`: coefficient `0.002060`, |coef| `0.002060`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001695`, |coef| `0.001695`
- `lag_00__CT_kills_last_3s`: coefficient `0.001670`, |coef| `0.001670`
- `lag_00__T3__has_bomb`: coefficient `-0.001428`, |coef| `0.001428`
- `lag_00__kill_diff_last_3s`: coefficient `0.001392`, |coef| `0.001392`
- `lag_08__CT2__duck_amount`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_06__T1__is_walking`: coefficient `-0.001250`, |coef| `0.001250`
- `lag_00__T3__alive`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_00__damage_diff_last_5s`: coefficient `0.001241`, |coef| `0.001241`
- `lag_00__CT_damage_last_5s`: coefficient `0.001208`, |coef| `0.001208`
- `lag_02__T_flashed_players`: coefficient `0.001149`, |coef| `0.001149`
- `lag_00__T3__armor`: coefficient `-0.001139`, |coef| `0.001139`
- `lag_01__CT_flashed_players`: coefficient `0.001131`, |coef| `0.001131`
- `lag_00__T3__hp`: coefficient `-0.001118`, |coef| `0.001118`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.003161` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001695` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001114` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.001100` (raises CT win probability)
- `lag_05__CT4__smoke`: coefficient `-0.001062` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.001051` (raises CT win probability)
- `lag_02__T5__molly`: coefficient `-0.001029` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001014` (raises CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `0.000825` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `-0.000821` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_flashed_players`: coefficient `0.002060` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001670` (raises CT win probability)
- `lag_00__T3__has_bomb`: coefficient `-0.001428` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001392` (raises CT win probability)
- `lag_08__CT2__duck_amount`: coefficient `-0.001271` (lowers CT win probability)
- `lag_06__T1__is_walking`: coefficient `-0.001250` (lowers CT win probability)
- `lag_00__T3__alive`: coefficient `-0.001242` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001241` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001208` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.001149` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `119479`, seconds `35.50`, LSTM delta `+0.1665`

Top all feature movements:
- `lag_00__CT2__flash_duration`: contribution `+0.022072`
- `lag_00__CT_flashed_players`: contribution `+0.013531`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007089`
- `lag_00__T_flashed_players`: contribution `+0.005555`
- `lag_08__CT2__duck_amount`: contribution `+0.004842`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.022072`
- `lag_00__CT_flash_duration_sum`: contribution `+0.007089`
- `lag_05__CT4__smoke`: contribution `+0.002317`
- `lag_02__T5__molly`: contribution `+0.002277`

### tick `119639`, seconds `38.00`, LSTM delta `+0.1037`

Top all feature movements:
- `lag_02__T5__flash_duration`: contribution `+0.008838`
- `lag_02__T1__flash_duration`: contribution `+0.008666`
- `lag_02__T_flash_duration_sum`: contribution `+0.007259`
- `lag_02__T_flashed_players`: contribution `+0.006653`
- `lag_05__CT2__flash_duration`: contribution `+0.005696`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.008838`
- `lag_02__T1__flash_duration`: contribution `+0.008666`
- `lag_02__T_flash_duration_sum`: contribution `+0.007259`
- `lag_05__CT2__flash_duration`: contribution `+0.005696`
- `lag_05__CT_flash_duration_sum`: contribution `+0.002103`

### tick `119735`, seconds `39.50`, LSTM delta `+0.0650`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004821`
- `lag_05__T_flashed_players`: contribution `+0.004404`
- `lag_05__T5__flash_duration`: contribution `+0.004061`
- `lag_00__T5__flash_duration`: contribution `+0.003801`
- `lag_05__T1__flash_duration`: contribution `+0.003396`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `+0.004061`
- `lag_00__T5__flash_duration`: contribution `+0.003801`
- `lag_05__T1__flash_duration`: contribution `+0.003396`
- `lag_05__T_flash_duration_sum`: contribution `+0.003202`
- `lag_08__CT2__flash_duration`: contribution `+0.002720`

### tick `119511`, seconds `36.00`, LSTM delta `+0.0542`

Top all feature movements:
- `lag_01__CT_flashed_players`: contribution `+0.007434`
- `lag_01__CT2__flash_duration`: contribution `+0.007339`
- `lag_01__T_flashed_players`: contribution `+0.005889`
- `lag_00__T_flashed_players`: contribution `-0.005555`
- `lag_00__CT_flashed_players`: contribution `-0.004510`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.007339`
- `lag_01__CT_flash_duration_sum`: contribution `+0.002734`
- `lag_00__T_B_site_active_infernos`: contribution `+0.001458`

### tick `119575`, seconds `37.00`, LSTM delta `-0.0459`

Top all feature movements:
- `lag_00__CT2__flash_duration`: contribution `-0.007844`
- `lag_02__T_flashed_players`: contribution `-0.006653`
- `lag_00__T_flashed_players`: contribution `+0.005555`
- `lag_00__CT_flashed_players`: contribution `+0.004510`
- `lag_03__CT_flashed_players`: contribution `-0.004451`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.007844`
- `lag_00__T5__flash_duration`: contribution `-0.003801`
- `lag_03__CT2__flash_duration`: contribution `-0.002634`
- `lag_00__T_flash_duration_sum`: contribution `-0.001857`
- `lag_03__CT_flash_duration_sum`: contribution `-0.001389`
