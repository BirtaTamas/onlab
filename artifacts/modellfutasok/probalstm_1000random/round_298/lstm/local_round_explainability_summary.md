# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX/flyquest-vs-fluxo-ancient.csv`
- round_num: `10`

## Largest probability jumps

- tick `63157`, seconds `38.50`, LSTM `0.7138`, delta `+0.1636`
- tick `64053`, seconds `52.50`, LSTM `0.8232`, delta `+0.1632`
- tick `62677`, seconds `31.00`, LSTM `0.5760`, delta `-0.1321`
- tick `64757`, seconds `63.50`, LSTM `0.9195`, delta `+0.1251`
- tick `65365`, seconds `73.00`, LSTM `0.9466`, delta `+0.1074`
- tick `63509`, seconds `44.00`, LSTM `0.6700`, delta `-0.0630`
- tick `64085`, seconds `53.00`, LSTM `0.8666`, delta `+0.0434`
- tick `65557`, seconds `76.00`, LSTM `0.9393`, delta `-0.0388`
- tick `63189`, seconds `39.00`, LSTM `0.7524`, delta `+0.0386`
- tick `61077`, seconds `6.00`, LSTM `0.7719`, delta `-0.0378`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002677`, |coef| `0.002677`
- `lag_00__CT_kills_last_3s`: coefficient `0.002413`, |coef| `0.002413`
- `lag_00__damage_diff_last_5s`: coefficient `0.002313`, |coef| `0.002313`
- `lag_00__CT_damage_last_5s`: coefficient `0.001793`, |coef| `0.001793`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.001580`, |coef| `0.001580`
- `lag_01__T3__is_walking`: coefficient `-0.001421`, |coef| `0.001421`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001300`, |coef| `0.001300`
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001221`, |coef| `0.001221`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `-0.001183`, |coef| `0.001183`
- `lag_01__CT_kills_last_3s`: coefficient `0.001180`, |coef| `0.001180`
- `lag_06__T_place_SIDEHALL`: coefficient `0.001152`, |coef| `0.001152`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001139`, |coef| `0.001139`
- `lag_06__CT2__is_walking`: coefficient `-0.001124`, |coef| `0.001124`
- `lag_07__CT3__duck_amount`: coefficient `-0.001091`, |coef| `0.001091`
- `lag_01__kill_diff_last_3s`: coefficient `0.001067`, |coef| `0.001067`

## Top 10 utility ridge features

- `lag_05__T_B_site_active_infernos`: coefficient `-0.000892` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000757` (lowers CT win probability)
- `lag_15__CT3__smoke`: coefficient `-0.000745` (lowers CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.000681` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000679` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.000657` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000620` (lowers CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.000615` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `0.000594` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `0.000588` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002677` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002413` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002313` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001793` (raises CT win probability)
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.001580` (lowers CT win probability)
- `lag_01__T3__is_walking`: coefficient `-0.001421` (lowers CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001300` (lowers CT win probability)
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001221` (lowers CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `-0.001183` (lowers CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.001180` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `63157`, seconds `38.50`, LSTM delta `+0.1636`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006967`
- `lag_00__kill_diff_last_3s`: contribution `+0.006444`
- `lag_00__damage_diff_last_5s`: contribution `+0.004905`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003957`
- `lag_00__CT_damage_last_5s`: contribution `+0.003674`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `64053`, seconds `52.50`, LSTM delta `+0.1632`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006967`
- `lag_00__kill_diff_last_3s`: contribution `+0.006444`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `+0.006360`
- `lag_00__damage_diff_last_5s`: contribution `+0.005218`
- `lag_07__T_place_TSIDELOWER`: contribution `+0.004576`

Top utility-only movements:
- `lag_05__T_B_site_active_infernos`: contribution `+0.002523`

### tick `62677`, seconds `31.00`, LSTM delta `-0.1321`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.006444`
- `lag_00__damage_diff_last_5s`: contribution `-0.004122`
- `lag_03__CT_place_SIDEENTRANCE`: contribution `-0.003936`
- `lag_06__CT_place_SIDEENTRANCE`: contribution `-0.003727`
- `lag_12__T1__flash_duration`: contribution `-0.003428`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `-0.003428`

### tick `64757`, seconds `63.50`, LSTM delta `+0.1251`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.008424`
- `lag_06__T_place_SIDEHALL`: contribution `+0.007465`
- `lag_00__CT_kills_last_3s`: contribution `+0.006967`
- `lag_00__kill_diff_last_3s`: contribution `+0.006444`
- `lag_07__CT5__is_scoped`: contribution `+0.003519`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65365`, seconds `73.00`, LSTM delta `+0.1074`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006967`
- `lag_00__kill_diff_last_3s`: contribution `+0.006444`
- `lag_07__T_place_SIDEHALL`: contribution `+0.005774`
- `lag_00__damage_diff_last_5s`: contribution `+0.005218`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003957`

Top utility-only movements:
- `lag_14__CT_B_site_active_infernos`: contribution `+0.002017`
- `lag_00__T2__smoke`: contribution `+0.001362`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001124`
