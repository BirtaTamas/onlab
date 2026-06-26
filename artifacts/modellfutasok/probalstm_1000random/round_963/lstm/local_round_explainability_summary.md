# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `26594`, seconds `95.50`, LSTM `0.4372`, delta `-0.3864`
- tick `26626`, seconds `96.00`, LSTM `0.1672`, delta `-0.2700`
- tick `26242`, seconds `90.00`, LSTM `0.7683`, delta `+0.2404`
- tick `26402`, seconds `92.50`, LSTM `0.9159`, delta `+0.1639`
- tick `26050`, seconds `87.00`, LSTM `0.5884`, delta `+0.1298`
- tick `26754`, seconds `98.00`, LSTM `0.0182`, delta `-0.0991`
- tick `26466`, seconds `93.50`, LSTM `0.8469`, delta `-0.0762`
- tick `23938`, seconds `54.00`, LSTM `0.4534`, delta `+0.0542`
- tick `24098`, seconds `56.50`, LSTM `0.5088`, delta `+0.0512`
- tick `25602`, seconds `80.00`, LSTM `0.4447`, delta `+0.0505`

## Top 15 local ridge features

- `lag_01__T_place_HOLE`: coefficient `0.003199`, |coef| `0.003199`
- `lag_11__T_place_HOLE`: coefficient `-0.002615`, |coef| `0.002615`
- `lag_06__T_place_HOLE`: coefficient `-0.002503`, |coef| `0.002503`
- `lag_07__T_place_HOLE`: coefficient `-0.002230`, |coef| `0.002230`
- `lag_07__T_place_BDOORS`: coefficient `0.002152`, |coef| `0.002152`
- `lag_00__kill_diff_last_3s`: coefficient `0.002135`, |coef| `0.002135`
- `lag_00__T_place_HOLE`: coefficient `0.001865`, |coef| `0.001865`
- `lag_00__CT_place_HOLE`: coefficient `0.001799`, |coef| `0.001799`
- `lag_04__CT3__flash_duration`: coefficient `0.001798`, |coef| `0.001798`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001781`, |coef| `0.001781`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_05__T_place_BDOORS`: coefficient `0.001702`, |coef| `0.001702`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001679`, |coef| `0.001679`
- `lag_12__T_place_HOLE`: coefficient `-0.001650`, |coef| `0.001650`
- `lag_10__CT3__flash_duration`: coefficient `0.001621`, |coef| `0.001621`

## Top 10 utility ridge features

- `lag_04__CT3__flash_duration`: coefficient `0.001798` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.001621` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001490` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001435` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.001329` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.001313` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.001295` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.001148` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.001000` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000965` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_HOLE`: coefficient `0.003199` (raises CT win probability)
- `lag_11__T_place_HOLE`: coefficient `-0.002615` (lowers CT win probability)
- `lag_06__T_place_HOLE`: coefficient `-0.002503` (lowers CT win probability)
- `lag_07__T_place_HOLE`: coefficient `-0.002230` (lowers CT win probability)
- `lag_07__T_place_BDOORS`: coefficient `0.002152` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002135` (raises CT win probability)
- `lag_00__T_place_HOLE`: coefficient `0.001865` (raises CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001799` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `-0.001781` (lowers CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001711` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26594`, seconds `95.50`, LSTM delta `-0.3864`

Top all feature movements:
- `lag_01__T_place_HOLE`: contribution `-0.082453`
- `lag_06__T_place_HOLE`: contribution `-0.064511`
- `lag_15__CT_shots_fired_sum`: contribution `-0.016423`
- `lag_12__T_place_BDOORS`: contribution `-0.013673`
- `lag_15__CT1__shots_fired`: contribution `-0.012032`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `-0.010075`
- `lag_04__CT1__flash_duration`: contribution `-0.009635`
- `lag_10__T3__flash_duration`: contribution `-0.005805`
- `lag_10__CT1__flash_duration`: contribution `-0.005600`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.004692`

### tick `26626`, seconds `96.00`, LSTM delta `-0.2700`

Top all feature movements:
- `lag_07__T_place_HOLE`: contribution `-0.057483`
- `lag_07__T_place_BDOORS`: contribution `-0.026915`
- `lag_02__T_place_HOLE`: contribution `-0.025010`
- `lag_11__T_place_BDOORS`: contribution `-0.012713`
- `lag_13__T_place_BDOORS`: contribution `-0.011307`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `-0.010215`
- `lag_11__CT1__flash_duration`: contribution `-0.005525`
- `lag_07__T3__flash_duration`: contribution `-0.003420`
- `lag_05__CT1__flash_duration`: contribution `-0.003251`
- `lag_11__T1__flash_duration`: contribution `-0.003176`

### tick `26242`, seconds `90.00`, LSTM delta `+0.2404`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.022278`
- `lag_05__T_place_BDOORS`: contribution `+0.021295`
- `lag_01__T_place_BDOORS`: contribution `+0.014881`
- `lag_04__CT1__shots_fired`: contribution `+0.014329`
- `lag_10__CT3__flash_duration`: contribution `+0.012613`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.012613`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.004692`

### tick `26402`, seconds `92.50`, LSTM delta `+0.1639`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.048084`
- `lag_04__T_place_BDOORS`: contribution `+0.015727`
- `lag_00__T_place_BDOORS`: contribution `+0.012874`
- `lag_10__T_place_BDOORS`: contribution `-0.011699`
- `lag_04__CT1__flash_duration`: contribution `+0.009635`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.009635`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004167`
- `lag_01__CT3__flash_duration`: contribution `+0.003635`

### tick `26050`, seconds `87.00`, LSTM delta `+0.1298`

Top all feature movements:
- `lag_04__CT3__flash_duration`: contribution `+0.013990`
- `lag_00__T_place_MIDDOORS`: contribution `+0.007272`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006997`
- `lag_00__kill_diff_last_3s`: contribution `+0.005139`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004993`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.013990`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004993`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.002499`
