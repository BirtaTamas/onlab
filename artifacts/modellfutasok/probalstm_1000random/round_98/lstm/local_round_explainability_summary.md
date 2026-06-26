# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `106789`, seconds `20.00`, LSTM `0.2475`, delta `-0.2877`
- tick `106853`, seconds `21.00`, LSTM `0.1370`, delta `-0.1755`
- tick `106885`, seconds `21.50`, LSTM `0.0413`, delta `-0.0958`
- tick `106821`, seconds `20.50`, LSTM `0.3125`, delta `+0.0650`
- tick `107237`, seconds `27.00`, LSTM `0.0873`, delta `+0.0632`
- tick `107557`, seconds `32.00`, LSTM `0.0224`, delta `-0.0597`
- tick `107301`, seconds `28.00`, LSTM `0.1300`, delta `+0.0302`
- tick `106661`, seconds `18.00`, LSTM `0.5576`, delta `+0.0272`
- tick `106341`, seconds `13.00`, LSTM `0.4911`, delta `-0.0269`
- tick `107333`, seconds `28.50`, LSTM `0.1552`, delta `+0.0252`

## Top 15 local ridge features

- `lag_15__CT_place_TRUCK`: coefficient `-0.001745`, |coef| `0.001745`
- `lag_01__T3__flash_duration`: coefficient `-0.001675`, |coef| `0.001675`
- `lag_09__CT_place_TRUCK`: coefficient `-0.001581`, |coef| `0.001581`
- `lag_00__T_kills_last_3s`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_03__T3__flash_duration`: coefficient `-0.001341`, |coef| `0.001341`
- `lag_00__T_damage_last_5s`: coefficient `-0.001251`, |coef| `0.001251`
- `lag_06__T2__flash_duration`: coefficient `0.001241`, |coef| `0.001241`
- `lag_14__T_flashed_players`: coefficient `-0.001223`, |coef| `0.001223`
- `lag_06__T1__flash_duration`: coefficient `0.001210`, |coef| `0.001210`
- `lag_11__T3__duck_amount`: coefficient `0.001200`, |coef| `0.001200`
- `lag_14__T2__flash_duration`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_00__damage_diff_last_5s`: coefficient `0.001178`, |coef| `0.001178`
- `lag_12__T3__duck_amount`: coefficient `-0.001162`, |coef| `0.001162`
- `lag_00__kill_diff_last_3s`: coefficient `0.001114`, |coef| `0.001114`
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001046`, |coef| `0.001046`

## Top 10 utility ridge features

- `lag_01__T3__flash_duration`: coefficient `-0.001675` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.001341` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.001241` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001210` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `-0.001190` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001046` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `-0.001033` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `-0.001029` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001028` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000987` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_TRUCK`: coefficient `-0.001745` (lowers CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `-0.001581` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001368` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001251` (lowers CT win probability)
- `lag_14__T_flashed_players`: coefficient `-0.001223` (lowers CT win probability)
- `lag_11__T3__duck_amount`: coefficient `0.001200` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001178` (raises CT win probability)
- `lag_12__T3__duck_amount`: coefficient `-0.001162` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001114` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001026` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `106789`, seconds `20.00`, LSTM delta `-0.2877`

Top all feature movements:
- `lag_15__CT_place_TRUCK`: contribution `-0.011256`
- `lag_09__CT_place_TRUCK`: contribution `-0.010195`
- `lag_01__T3__flash_duration`: contribution `-0.010031`
- `lag_14__T_flashed_players`: contribution `-0.007079`
- `lag_06__T2__flash_duration`: contribution `-0.005538`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.010031`
- `lag_06__T2__flash_duration`: contribution `-0.005538`
- `lag_14__T2__flash_duration`: contribution `-0.005308`
- `lag_06__T1__flash_duration`: contribution `-0.005281`
- `lag_14__T_flash_duration_sum`: contribution `-0.004891`

### tick `106853`, seconds `21.00`, LSTM delta `-0.1755`

Top all feature movements:
- `lag_03__T3__flash_duration`: contribution `-0.008032`
- `lag_11__T3__duck_amount`: contribution `-0.004524`
- `lag_12__T3__duck_amount`: contribution `-0.004381`
- `lag_08__T2__flash_duration`: contribution `-0.004310`
- `lag_08__T1__flash_duration`: contribution `-0.004168`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.008032`
- `lag_08__T2__flash_duration`: contribution `-0.004310`
- `lag_08__T1__flash_duration`: contribution `-0.004168`
- `lag_08__T_flash_duration_sum`: contribution `-0.002848`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.002636`

### tick `106885`, seconds `21.50`, LSTM delta `-0.0958`

Top all feature movements:
- `lag_04__T3__flash_duration`: contribution `-0.006186`
- `lag_12__CT_place_TRUCK`: contribution `-0.005821`
- `lag_12__T3__duck_amount`: contribution `+0.004381`
- `lag_00__T_kills_last_3s`: contribution `-0.004333`
- `lag_09__T_flashed_players`: contribution `-0.003910`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `-0.006186`
- `lag_09__T_flash_duration_sum`: contribution `-0.002631`
- `lag_09__T2__flash_duration`: contribution `-0.002457`
- `lag_09__T1__flash_duration`: contribution `-0.002406`

### tick `106821`, seconds `20.50`, LSTM delta `+0.0650`

Top all feature movements:
- `lag_12__CT_place_TRUCK`: contribution `+0.005821`
- `lag_11__T3__duck_amount`: contribution `+0.004524`
- `lag_12__T3__duck_amount`: contribution `+0.004381`
- `lag_06__T3__duck_amount`: contribution `+0.002832`
- `lag_10__T3__duck_amount`: contribution `+0.002639`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.001945`

### tick `107237`, seconds `27.00`, LSTM delta `+0.0632`

Top all feature movements:
- `lag_04__T3__flash_duration`: contribution `+0.006186`
- `lag_00__T_A_site_active_infernos`: contribution `+0.003059`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002777`
- `lag_10__T1__shots_fired`: contribution `+0.002756`
- `lag_00__kill_diff_last_3s`: contribution `+0.002681`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.006186`
- `lag_00__T_A_site_active_infernos`: contribution `+0.003059`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002777`
- `lag_11__T_active_infernos`: contribution `+0.001611`
- `lag_10__CT2__flash_duration`: contribution `+0.001495`
