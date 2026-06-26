# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `42169`, seconds `33.00`, LSTM `0.8610`, delta `+0.1752`
- tick `42073`, seconds `31.50`, LSTM `0.5306`, delta `-0.1362`
- tick `45465`, seconds `84.50`, LSTM `0.9249`, delta `+0.1303`
- tick `42137`, seconds `32.50`, LSTM `0.6858`, delta `+0.1069`
- tick `42105`, seconds `32.00`, LSTM `0.5788`, delta `+0.0482`
- tick `41785`, seconds `27.00`, LSTM `0.6119`, delta `+0.0422`
- tick `43961`, seconds `61.00`, LSTM `0.8232`, delta `-0.0302`
- tick `41977`, seconds `30.00`, LSTM `0.6736`, delta `+0.0280`
- tick `41305`, seconds `19.50`, LSTM `0.5903`, delta `+0.0223`
- tick `44121`, seconds `63.50`, LSTM `0.8608`, delta `+0.0218`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002639`, |coef| `0.002639`
- `lag_00__kill_diff_last_3s`: coefficient `0.002495`, |coef| `0.002495`
- `lag_00__damage_diff_last_5s`: coefficient `0.002379`, |coef| `0.002379`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002265`, |coef| `0.002265`
- `lag_00__CT_damage_last_5s`: coefficient `0.002002`, |coef| `0.002002`
- `lag_06__T2__is_walking`: coefficient `-0.001995`, |coef| `0.001995`
- `lag_00__CT4__duck_amount`: coefficient `0.001845`, |coef| `0.001845`
- `lag_00__T2__alive`: coefficient `-0.001717`, |coef| `0.001717`
- `lag_00__T2__hp`: coefficient `-0.001696`, |coef| `0.001696`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001679`, |coef| `0.001679`
- `lag_00__T2__armor`: coefficient `-0.001632`, |coef| `0.001632`
- `lag_00__T2__molly`: coefficient `-0.001595`, |coef| `0.001595`
- `lag_00__T2__smoke`: coefficient `-0.001572`, |coef| `0.001572`
- `lag_00__T_macro_B`: coefficient `-0.001501`, |coef| `0.001501`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001501`, |coef| `0.001501`

## Top 10 utility ridge features

- `lag_00__T2__molly`: coefficient `-0.001595` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.001572` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001381` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001368` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001226` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001170` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000852` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000847` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000846` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `-0.000814` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002639` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002495` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002379` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002265` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002002` (raises CT win probability)
- `lag_06__T2__is_walking`: coefficient `-0.001995` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.001845` (raises CT win probability)
- `lag_00__T2__alive`: coefficient `-0.001717` (lowers CT win probability)
- `lag_00__T2__hp`: coefficient `-0.001696` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001679` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `42169`, seconds `33.00`, LSTM delta `+0.1752`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.018880`
- `lag_00__CT_kills_last_3s`: contribution `+0.015239`
- `lag_00__kill_diff_last_3s`: contribution `+0.012013`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010496`
- `lag_00__T1__flash_duration`: contribution `+0.008188`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.008188`
- `lag_00__T5__flash_duration`: contribution `+0.008064`
- `lag_14__T3__flash_duration`: contribution `+0.006209`
- `lag_00__T_flash_duration_sum`: contribution `+0.005690`
- `lag_10__CT4__flash_duration`: contribution `+0.005535`

### tick `42073`, seconds `31.50`, LSTM delta `-0.1362`

Top all feature movements:
- `lag_00__T1__flash_duration`: contribution `-0.008188`
- `lag_00__T5__flash_duration`: contribution `-0.008064`
- `lag_11__T3__flash_duration`: contribution `-0.006172`
- `lag_00__kill_diff_last_3s`: contribution `-0.006006`
- `lag_07__CT4__flash_duration`: contribution `-0.005988`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.008188`
- `lag_00__T5__flash_duration`: contribution `-0.008064`
- `lag_11__T3__flash_duration`: contribution `-0.006172`
- `lag_07__CT4__flash_duration`: contribution `-0.005988`
- `lag_07__CT3__flash_duration`: contribution `-0.005368`

### tick `45465`, seconds `84.50`, LSTM delta `+0.1303`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012587`
- `lag_00__CT_kills_last_3s`: contribution `+0.007619`
- `lag_00__CT4__duck_amount`: contribution `+0.006775`
- `lag_00__kill_diff_last_3s`: contribution `+0.006006`
- `lag_00__damage_diff_last_5s`: contribution `+0.005367`

Top utility-only movements:
- `lag_00__T2__molly`: contribution `+0.003554`
- `lag_00__T2__smoke`: contribution `+0.003453`

### tick `42137`, seconds `32.50`, LSTM delta `+0.1069`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014160`
- `lag_00__damage_diff_last_5s`: contribution `+0.006119`
- `lag_13__T3__flash_duration`: contribution `+0.005317`
- `lag_00__CT_damage_last_5s`: contribution `+0.004976`
- `lag_09__CT4__flash_duration`: contribution `+0.004252`

Top utility-only movements:
- `lag_13__T3__flash_duration`: contribution `+0.005317`
- `lag_09__CT4__flash_duration`: contribution `+0.004252`
- `lag_09__CT3__flash_duration`: contribution `+0.003950`
- `lag_09__CT_flash_duration_sum`: contribution `+0.003103`
- `lag_02__T1__flash_duration`: contribution `+0.002812`

### tick `42105`, seconds `32.00`, LSTM delta `+0.0482`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007619`
- `lag_00__kill_diff_last_3s`: contribution `+0.006006`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004720`
- `lag_00__damage_diff_last_5s`: contribution `-0.004401`
- `lag_00__CT_damage_last_5s`: contribution `-0.003972`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `+0.003198`
- `lag_08__CT3__flash_duration`: contribution `+0.002156`
- `lag_08__CT4__flash_duration`: contribution `+0.002100`
- `lag_08__CT_flash_duration_sum`: contribution `+0.001558`
- `lag_12__T3__flash_duration`: contribution `+0.001401`
