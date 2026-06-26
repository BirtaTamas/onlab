# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `57215`, seconds `97.50`, LSTM `0.7357`, delta `+0.2047`
- tick `57343`, seconds `99.50`, LSTM `0.3685`, delta `-0.1937`
- tick `57183`, seconds `97.00`, LSTM `0.5310`, delta `+0.1767`
- tick `57407`, seconds `100.50`, LSTM `0.2234`, delta `-0.1411`
- tick `57247`, seconds `98.00`, LSTM `0.6107`, delta `-0.1250`
- tick `57119`, seconds `96.00`, LSTM `0.3513`, delta `-0.0776`
- tick `57471`, seconds `101.50`, LSTM `0.1899`, delta `-0.0514`
- tick `53535`, seconds `40.00`, LSTM `0.4776`, delta `+0.0420`
- tick `58303`, seconds `114.50`, LSTM `0.0121`, delta `-0.0377`
- tick `57503`, seconds `102.00`, LSTM `0.1534`, delta `-0.0365`

## Top 15 local ridge features

- `lag_00__T_flashed_players`: coefficient `-0.002726`, |coef| `0.002726`
- `lag_00__kill_diff_last_3s`: coefficient `0.002628`, |coef| `0.002628`
- `lag_01__T_flashed_players`: coefficient `-0.002126`, |coef| `0.002126`
- `lag_06__CT2__flash_duration`: coefficient `0.002089`, |coef| `0.002089`
- `lag_00__CT3__flash_duration`: coefficient `0.002067`, |coef| `0.002067`
- `lag_00__CT_kills_last_3s`: coefficient `0.002057`, |coef| `0.002057`
- `lag_01__T4__duck_amount`: coefficient `0.001686`, |coef| `0.001686`
- `lag_00__damage_diff_last_5s`: coefficient `0.001550`, |coef| `0.001550`
- `lag_07__CT_place_ARAMP`: coefficient `0.001540`, |coef| `0.001540`
- `lag_07__CT2__flash_duration`: coefficient `0.001513`, |coef| `0.001513`
- `lag_09__CT_place_ARAMP`: coefficient `-0.001469`, |coef| `0.001469`
- `lag_10__CT_place_ARAMP`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_00__CT_flashed_players`: coefficient `0.001413`, |coef| `0.001413`
- `lag_03__T_flashed_players`: coefficient `0.001395`, |coef| `0.001395`
- `lag_09__CT_place_LONGA`: coefficient `0.001381`, |coef| `0.001381`

## Top 10 utility ridge features

- `lag_06__CT2__flash_duration`: coefficient `0.002089` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.002067` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.001513` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001274` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001155` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001138` (lowers CT win probability)
- `lag_08__T_active_infernos`: coefficient `0.001125` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.001003` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000982` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000967` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_flashed_players`: coefficient `-0.002726` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002628` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `-0.002126` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002057` (raises CT win probability)
- `lag_01__T4__duck_amount`: coefficient `0.001686` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001550` (raises CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `0.001540` (raises CT win probability)
- `lag_09__CT_place_ARAMP`: coefficient `-0.001469` (lowers CT win probability)
- `lag_10__CT_place_ARAMP`: coefficient `-0.001434` (lowers CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.001413` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `57215`, seconds `97.50`, LSTM delta `+0.2047`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `+0.012477`
- `lag_03__T_flashed_players`: contribution `+0.010769`
- `lag_00__CT_flashed_players`: contribution `+0.009283`
- `lag_10__CT_place_ARAMP`: contribution `+0.008935`
- `lag_01__T_flashed_players`: contribution `+0.008205`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.012477`
- `lag_07__CT2__flash_duration`: contribution `+0.007376`
- `lag_00__CT2__flash_duration`: contribution `+0.004381`

### tick `57343`, seconds `99.50`, LSTM delta `-0.1937`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.008916`
- `lag_02__T_shots_fired_sum`: contribution `-0.007265`
- `lag_06__T_flashed_players`: contribution `-0.006801`
- `lag_00__kill_diff_last_3s`: contribution `-0.006325`
- `lag_14__CT_place_ARAMP`: contribution `-0.006074`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.006055`
- `lag_11__CT2__flash_duration`: contribution `-0.004580`

### tick `57183`, seconds `97.00`, LSTM delta `+0.1767`

Top all feature movements:
- `lag_01__T_flashed_players`: contribution `+0.012307`
- `lag_00__T_flashed_players`: contribution `+0.010519`
- `lag_06__CT2__flash_duration`: contribution `+0.010186`
- `lag_09__CT_place_ARAMP`: contribution `+0.009148`
- `lag_00__kill_diff_last_3s`: contribution `+0.006325`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.010186`
- `lag_05__CT4__flash_duration`: contribution `+0.003842`
- `lag_08__T_B_site_active_infernos`: contribution `+0.002778`

### tick `57407`, seconds `100.50`, LSTM delta `-0.1411`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `-0.012477`
- `lag_03__T_shots_fired_sum`: contribution `-0.008024`
- `lag_06__CT2__flash_duration`: contribution `-0.007923`
- `lag_08__T_flashed_players`: contribution `-0.007824`
- `lag_00__kill_diff_last_3s`: contribution `-0.006325`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.012477`
- `lag_06__CT2__flash_duration`: contribution `-0.007923`
- `lag_06__CT3__flash_duration`: contribution `-0.003231`

### tick `57247`, seconds `98.00`, LSTM delta `-0.1250`

Top all feature movements:
- `lag_03__T_flashed_players`: contribution `-0.008076`
- `lag_01__CT_flashed_players`: contribution `-0.007606`
- `lag_00__kill_diff_last_3s`: contribution `-0.006325`
- `lag_11__CT_place_ARAMP`: contribution `-0.005465`
- `lag_00__T_shots_fired_sum`: contribution `+0.004562`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.003487`
