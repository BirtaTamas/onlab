# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `30877`, seconds `22.50`, LSTM `0.7657`, delta `+0.1521`
- tick `31165`, seconds `27.00`, LSTM `0.8894`, delta `+0.0933`
- tick `31197`, seconds `27.50`, LSTM `0.9540`, delta `+0.0646`
- tick `31133`, seconds `26.50`, LSTM `0.7961`, delta `+0.0633`
- tick `31037`, seconds `25.00`, LSTM `0.7259`, delta `-0.0508`
- tick `35165`, seconds `89.50`, LSTM `0.9659`, delta `+0.0456`
- tick `34909`, seconds `85.50`, LSTM `0.8921`, delta `-0.0314`
- tick `29789`, seconds `5.50`, LSTM `0.6166`, delta `-0.0307`
- tick `35101`, seconds `88.50`, LSTM `0.9226`, delta `+0.0288`
- tick `29885`, seconds `7.00`, LSTM `0.5899`, delta `-0.0254`

## Top 15 local ridge features

- `lag_11__CT_place_SQUEAKY`: coefficient `0.001302`, |coef| `0.001302`
- `lag_13__CT_place_CONTROL`: coefficient `0.001126`, |coef| `0.001126`
- `lag_00__kill_diff_last_3s`: coefficient `0.001021`, |coef| `0.001021`
- `lag_04__CT_place_LOCKERROOM`: coefficient `0.000873`, |coef| `0.000873`
- `lag_00__CT_kills_last_3s`: coefficient `0.000836`, |coef| `0.000836`
- `lag_03__CT_place_LOCKERROOM`: coefficient `0.000836`, |coef| `0.000836`
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.000804`, |coef| `0.000804`
- `lag_00__T_mollies_last_5s`: coefficient `0.000763`, |coef| `0.000763`
- `lag_10__CT_place_HUTROOF`: coefficient `0.000754`, |coef| `0.000754`
- `lag_03__T_place_SILO`: coefficient `-0.000748`, |coef| `0.000748`
- `lag_00__CT_place_TUNNELS`: coefficient `0.000738`, |coef| `0.000738`
- `lag_01__T_place_TROPHY`: coefficient `-0.000730`, |coef| `0.000730`
- `lag_10__CT_place_RAFTERS`: coefficient `-0.000711`, |coef| `0.000711`
- `lag_13__T_place_VENDING`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_00__T2__shots_fired`: coefficient `0.000690`, |coef| `0.000690`

## Top 10 utility ridge features

- `lag_00__T_mollies_last_5s`: coefficient `0.000763` (raises CT win probability)
- `lag_10__T_mollies_last_5s`: coefficient `-0.000428` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000395` (lowers CT win probability)
- `lag_13__T_mollies_last_5s`: coefficient `-0.000384` (lowers CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `0.000381` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000376` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000369` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.000368` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000365` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000353` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_SQUEAKY`: coefficient `0.001302` (raises CT win probability)
- `lag_13__CT_place_CONTROL`: coefficient `0.001126` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001021` (raises CT win probability)
- `lag_04__CT_place_LOCKERROOM`: coefficient `0.000873` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000836` (raises CT win probability)
- `lag_03__CT_place_LOCKERROOM`: coefficient `0.000836` (raises CT win probability)
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.000804` (lowers CT win probability)
- `lag_10__CT_place_HUTROOF`: coefficient `0.000754` (raises CT win probability)
- `lag_03__T_place_SILO`: coefficient `-0.000748` (lowers CT win probability)
- `lag_00__CT_place_TUNNELS`: coefficient `0.000738` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `30877`, seconds `22.50`, LSTM delta `+0.1521`

Top all feature movements:
- `lag_11__CT_place_SQUEAKY`: contribution `+0.017312`
- `lag_13__CT_place_CONTROL`: contribution `+0.011688`
- `lag_10__CT_place_HUTROOF`: contribution `+0.005273`
- `lag_03__T_place_SILO`: contribution `+0.005084`
- `lag_04__T_place_CONTROL`: contribution `+0.004865`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `+0.001687`

### tick `31165`, seconds `27.00`, LSTM delta `+0.0933`

Top all feature movements:
- `lag_04__CT_place_LOCKERROOM`: contribution `+0.010871`
- `lag_13__T_place_CONTROL`: contribution `+0.004026`
- `lag_10__T_place_CONTROL`: contribution `+0.003801`
- `lag_10__T_place_TROPHY`: contribution `+0.003681`
- `lag_13__T_place_TROPHY`: contribution `-0.003525`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `+0.002024`
- `lag_04__CT5__flash_duration`: contribution `+0.001741`

### tick `31197`, seconds `27.50`, LSTM delta `+0.0646`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `+0.010005`
- `lag_05__CT_place_LOCKERROOM`: contribution `+0.005217`
- `lag_11__T_place_CONTROL`: contribution `+0.003499`
- `lag_00__kill_diff_last_3s`: contribution `+0.002459`
- `lag_00__CT_kills_last_3s`: contribution `+0.002414`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.001687`
- `lag_15__CT5__flash_duration`: contribution `+0.001301`

### tick `31133`, seconds `26.50`, LSTM delta `+0.0633`

Top all feature movements:
- `lag_03__CT_place_LOCKERROOM`: contribution `+0.010402`
- `lag_13__T_place_TROPHY`: contribution `+0.003525`
- `lag_13__T_place_VENDING`: contribution `+0.003521`
- `lag_11__T_place_SILO`: contribution `+0.002764`
- `lag_12__T_place_CONTROL`: contribution `+0.002579`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.001827`
- `lag_03__CT5__flash_duration`: contribution `+0.001253`

### tick `31037`, seconds `25.00`, LSTM delta `-0.0508`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.010005`
- `lag_10__T_place_TROPHY`: contribution `-0.003681`
- `lag_08__T_place_SILO`: contribution `-0.002344`
- `lag_15__T1__duck_amount`: contribution `-0.002153`
- `lag_15__CT_place_HUTROOF`: contribution `-0.002025`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.001847`
- `lag_10__CT5__flash_duration`: contribution `-0.001263`
