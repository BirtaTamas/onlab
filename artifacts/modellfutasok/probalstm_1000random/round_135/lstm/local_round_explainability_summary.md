# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `129537`, seconds `22.50`, LSTM `0.3081`, delta `-0.1758`
- tick `131297`, seconds `50.00`, LSTM `0.0800`, delta `-0.1296`
- tick `129569`, seconds `23.00`, LSTM `0.2430`, delta `-0.0651`
- tick `128897`, seconds `12.50`, LSTM `0.4355`, delta `-0.0480`
- tick `130113`, seconds `31.50`, LSTM `0.2398`, delta `+0.0440`
- tick `131041`, seconds `46.00`, LSTM `0.1789`, delta `-0.0428`
- tick `131265`, seconds `49.50`, LSTM `0.2096`, delta `-0.0419`
- tick `131009`, seconds `45.50`, LSTM `0.2218`, delta `+0.0417`
- tick `129601`, seconds `23.50`, LSTM `0.2044`, delta `-0.0386`
- tick `130753`, seconds `41.50`, LSTM `0.2080`, delta `-0.0368`

## Top 15 local ridge features

- `lag_04__CT2__flash_duration`: coefficient `-0.001617`, |coef| `0.001617`
- `lag_00__T_flashes_last_5s`: coefficient `-0.001601`, |coef| `0.001601`
- `lag_05__CT2__flash_duration`: coefficient `-0.001445`, |coef| `0.001445`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001182`, |coef| `0.001182`
- `lag_00__CT1__flash_duration`: coefficient `0.001122`, |coef| `0.001122`
- `lag_02__T_flashes_last_5s`: coefficient `-0.001066`, |coef| `0.001066`
- `lag_00__T_kills_last_3s`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_00__CT_place_BANANA`: coefficient `0.001010`, |coef| `0.001010`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001009`, |coef| `0.001009`
- `lag_04__CT1__flash_duration`: coefficient `-0.001001`, |coef| `0.001001`
- `lag_07__T_flashes_last_5s`: coefficient `0.000968`, |coef| `0.000968`
- `lag_03__CT2__flash_duration`: coefficient `-0.000947`, |coef| `0.000947`
- `lag_04__T_flashed_players`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000912`, |coef| `0.000912`

## Top 10 utility ridge features

- `lag_04__CT2__flash_duration`: coefficient `-0.001617` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001601` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001445` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001122` (raises CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.001066` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001014` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001009` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001001` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `0.000968` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.000947` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001182` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001059` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001010` (raises CT win probability)
- `lag_04__T_flashed_players`: coefficient `-0.000931` (lowers CT win probability)
- `lag_04__CT1__duck_amount`: coefficient `0.000873` (raises CT win probability)
- `lag_04__T2__is_walking`: coefficient `0.000811` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000805` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.000777` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `0.000774` (raises CT win probability)
- `lag_01__T_burning_players`: coefficient `-0.000767` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `129537`, seconds `22.50`, LSTM delta `-0.1758`

Top all feature movements:
- `lag_04__CT2__flash_duration`: contribution `-0.008652`
- `lag_00__CT_place_BALCONY`: contribution `-0.007588`
- `lag_00__CT1__flash_duration`: contribution `-0.006938`
- `lag_04__CT1__flash_duration`: contribution `-0.005349`
- `lag_04__CT_flash_duration_sum`: contribution `-0.004898`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `-0.008652`
- `lag_00__CT1__flash_duration`: contribution `-0.006938`
- `lag_04__CT1__flash_duration`: contribution `-0.005349`
- `lag_04__CT_flash_duration_sum`: contribution `-0.004898`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002802`

### tick `131297`, seconds `50.00`, LSTM delta `-0.1296`

Top all feature movements:
- `lag_05__CT2__flash_duration`: contribution `-0.010917`
- `lag_07__T_flashes_last_5s`: contribution `-0.008773`
- `lag_05__CT_flash_duration_sum`: contribution `-0.004424`
- `lag_05__T_flashed_players`: contribution `-0.003909`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003461`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `-0.010917`
- `lag_07__T_flashes_last_5s`: contribution `-0.008773`
- `lag_05__CT_flash_duration_sum`: contribution `-0.004424`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003461`
- `lag_00__CT2__flash_duration`: contribution `-0.003073`

### tick `129569`, seconds `23.00`, LSTM delta `-0.0651`

Top all feature movements:
- `lag_05__CT2__flash_duration`: contribution `-0.007728`
- `lag_01__CT_place_BALCONY`: contribution `-0.004414`
- `lag_05__CT_flash_duration_sum`: contribution `-0.004403`
- `lag_05__CT1__flash_duration`: contribution `-0.003057`
- `lag_01__T_flashed_players`: contribution `+0.002985`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `-0.007728`
- `lag_05__CT_flash_duration_sum`: contribution `-0.004403`
- `lag_05__CT1__flash_duration`: contribution `-0.003057`
- `lag_01__CT1__flash_duration`: contribution `-0.002355`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.001990`

### tick `128897`, seconds `12.50`, LSTM delta `-0.0480`

Top all feature movements:
- `lag_15__T_place_LOWERMID`: contribution `-0.003620`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.002639`
- `lag_15__CT_place_LIBRARY`: contribution `-0.002447`
- `lag_04__T_flashed_players`: contribution `-0.001796`
- `lag_03__CT_place_TOPOFMID`: contribution `-0.001648`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.001265`
- `lag_04__T1__flash_duration`: contribution `-0.001264`

### tick `130113`, seconds `31.50`, LSTM delta `+0.0440`

Top all feature movements:
- `lag_12__CT2__flash_duration`: contribution `+0.004259`
- `lag_07__T5__is_scoped`: contribution `+0.002434`
- `lag_15__T5__is_scoped`: contribution `+0.002053`
- `lag_04__T2__is_walking`: contribution `+0.001864`
- `lag_15__T1__duck_amount`: contribution `+0.001771`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `+0.004259`
- `lag_12__CT_active_infernos`: contribution `+0.000816`
