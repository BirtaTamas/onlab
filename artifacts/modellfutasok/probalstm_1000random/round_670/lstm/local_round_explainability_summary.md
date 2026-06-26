# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `15`

## Largest probability jumps

- tick `113736`, seconds `50.00`, LSTM `0.8437`, delta `+0.2097`
- tick `114344`, seconds `59.50`, LSTM `0.8398`, delta `+0.1326`
- tick `113928`, seconds `53.00`, LSTM `0.8008`, delta `-0.1013`
- tick `113000`, seconds `38.50`, LSTM `0.5416`, delta `-0.0927`
- tick `113704`, seconds `49.50`, LSTM `0.6340`, delta `+0.0919`
- tick `113416`, seconds `45.00`, LSTM `0.6216`, delta `+0.0645`
- tick `113064`, seconds `39.50`, LSTM `0.4785`, delta `-0.0616`
- tick `114056`, seconds `55.00`, LSTM `0.6986`, delta `-0.0539`
- tick `113544`, seconds `47.00`, LSTM `0.6264`, delta `-0.0430`
- tick `112776`, seconds `35.00`, LSTM `0.6094`, delta `+0.0421`

## Top 15 local ridge features

- `lag_06__CT1__flash_duration`: coefficient `0.001650`, |coef| `0.001650`
- `lag_15__CT_shots_fired_sum`: coefficient `-0.001188`, |coef| `0.001188`
- `lag_10__CT4__flash_duration`: coefficient `-0.001160`, |coef| `0.001160`
- `lag_06__CT_flashed_players`: coefficient `0.001160`, |coef| `0.001160`
- `lag_00__CT1__flash_duration`: coefficient `-0.001148`, |coef| `0.001148`
- `lag_00__CT4__flash_duration`: coefficient `-0.001145`, |coef| `0.001145`
- `lag_06__CT_flash_duration_sum`: coefficient `0.001145`, |coef| `0.001145`
- `lag_00__kill_diff_last_3s`: coefficient `0.001118`, |coef| `0.001118`
- `lag_01__CT1__flash_duration`: coefficient `-0.001030`, |coef| `0.001030`
- `lag_12__CT1__flash_duration`: coefficient `-0.001026`, |coef| `0.001026`
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.001017`, |coef| `0.001017`
- `lag_00__CT_kills_last_3s`: coefficient `0.000998`, |coef| `0.000998`
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000997`, |coef| `0.000997`
- `lag_01__CT_kills_last_3s`: coefficient `0.000986`, |coef| `0.000986`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000986`, |coef| `0.000986`

## Top 10 utility ridge features

- `lag_06__CT1__flash_duration`: coefficient `0.001650` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001160` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001148` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001145` (lowers CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001145` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.001030` (lowers CT win probability)
- `lag_12__CT1__flash_duration`: coefficient `-0.001026` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000997` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000903` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000895` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_shots_fired_sum`: coefficient `-0.001188` (lowers CT win probability)
- `lag_06__CT_flashed_players`: coefficient `0.001160` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001118` (raises CT win probability)
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.001017` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000998` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.000986` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.000986` (lowers CT win probability)
- `lag_15__CT4__is_scoped`: coefficient `0.000985` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000971` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.000958` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113736`, seconds `50.00`, LSTM delta `+0.2097`

Top all feature movements:
- `lag_06__CT1__flash_duration`: contribution `+0.009827`
- `lag_10__CT4__flash_duration`: contribution `+0.008348`
- `lag_06__CT_flashed_players`: contribution `+0.007622`
- `lag_06__CT_flash_duration_sum`: contribution `+0.006252`
- `lag_12__CT1__flash_duration`: contribution `+0.006169`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.009827`
- `lag_10__CT4__flash_duration`: contribution `+0.008348`
- `lag_06__CT_flash_duration_sum`: contribution `+0.006252`
- `lag_12__CT1__flash_duration`: contribution `+0.006169`
- `lag_08__T3__flash_duration`: contribution `+0.005640`

### tick `114344`, seconds `59.50`, LSTM delta `+0.1326`

Top all feature movements:
- `lag_15__CT_shots_fired_sum`: contribution `+0.016512`
- `lag_15__CT2__shots_fired`: contribution `+0.009718`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.004888`
- `lag_15__CT_place_ARAMP`: contribution `+0.004447`
- `lag_07__T1__flash_duration`: contribution `+0.004058`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `+0.004058`
- `lag_07__T_flash_duration_sum`: contribution `+0.001508`

### tick `113928`, seconds `53.00`, LSTM delta `-0.1013`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.009498`
- `lag_12__CT1__flash_duration`: contribution `-0.006114`
- `lag_00__kill_diff_last_3s`: contribution `-0.005381`
- `lag_02__CT2__shots_fired`: contribution `-0.005193`
- `lag_07__T1__flash_duration`: contribution `-0.004651`

Top utility-only movements:
- `lag_12__CT1__flash_duration`: contribution `-0.006114`
- `lag_07__T1__flash_duration`: contribution `-0.004651`
- `lag_00__CT4__flash_duration`: contribution `+0.004304`
- `lag_07__T_flash_duration_sum`: contribution `-0.003381`
- `lag_07__CT1__flash_duration`: contribution `-0.003351`

### tick `113000`, seconds `38.50`, LSTM delta `-0.0927`

Top all feature movements:
- `lag_06__CT1__flash_duration`: contribution `-0.009491`
- `lag_00__CT4__flash_duration`: contribution `-0.008237`
- `lag_00__CT1__flash_duration`: contribution `-0.006899`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005903`
- `lag_11__CT5__duck_amount`: contribution `-0.003599`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `-0.009491`
- `lag_00__CT4__flash_duration`: contribution `-0.008237`
- `lag_00__CT1__flash_duration`: contribution `-0.006899`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005903`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002959`

### tick `113704`, seconds `49.50`, LSTM delta `+0.0919`

Top all feature movements:
- `lag_09__CT4__flash_duration`: contribution `+0.006441`
- `lag_07__T3__flash_duration`: contribution `+0.004858`
- `lag_00__CT1__flash_duration`: contribution `+0.004530`
- `lag_05__CT1__flash_duration`: contribution `+0.004391`
- `lag_07__T_place_SHORTSTAIRS`: contribution `+0.004276`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `+0.006441`
- `lag_07__T3__flash_duration`: contribution `+0.004858`
- `lag_00__CT1__flash_duration`: contribution `+0.004530`
- `lag_05__CT1__flash_duration`: contribution `+0.004391`
- `lag_05__CT4__flash_duration`: contribution `+0.003920`
