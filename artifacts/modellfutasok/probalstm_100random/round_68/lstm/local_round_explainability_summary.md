# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `90414`, seconds `84.50`, LSTM `0.5617`, delta `+0.3991`
- tick `89582`, seconds `71.50`, LSTM `0.3454`, delta `+0.3049`
- tick `86894`, seconds `29.50`, LSTM `0.3238`, delta `-0.1937`
- tick `88750`, seconds `58.50`, LSTM `0.0749`, delta `-0.1810`
- tick `89614`, seconds `72.00`, LSTM `0.2610`, delta `-0.0844`
- tick `90638`, seconds `88.00`, LSTM `0.6686`, delta `+0.0532`
- tick `85678`, seconds `10.50`, LSTM `0.4858`, delta `+0.0519`
- tick `88718`, seconds `58.00`, LSTM `0.2559`, delta `-0.0486`
- tick `87822`, seconds `44.00`, LSTM `0.3290`, delta `+0.0482`
- tick `91182`, seconds `96.50`, LSTM `0.6646`, delta `-0.0468`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003789`, |coef| `0.003789`
- `lag_00__damage_diff_last_5s`: coefficient `0.003530`, |coef| `0.003530`
- `lag_11__T1__duck_amount`: coefficient `0.003483`, |coef| `0.003483`
- `lag_03__CT_place_CONNECTOR`: coefficient `-0.003203`, |coef| `0.003203`
- `lag_00__CT_kills_last_3s`: coefficient `0.003143`, |coef| `0.003143`
- `lag_07__T1__duck_amount`: coefficient `-0.003050`, |coef| `0.003050`
- `lag_03__CT_place_MIDDLE`: coefficient `0.002971`, |coef| `0.002971`
- `lag_00__T1__alive`: coefficient `-0.002917`, |coef| `0.002917`
- `lag_00__T1__hp`: coefficient `-0.002871`, |coef| `0.002871`
- `lag_00__CT4__is_walking`: coefficient `-0.002820`, |coef| `0.002820`
- `lag_15__CT_place_SHOP`: coefficient `0.002761`, |coef| `0.002761`
- `lag_00__T1__molly`: coefficient `-0.002668`, |coef| `0.002668`
- `lag_07__T_duck_amount_mean`: coefficient `-0.002604`, |coef| `0.002604`
- `lag_00__T1__has_helmet`: coefficient `-0.002510`, |coef| `0.002510`
- `lag_11__T_duck_amount_mean`: coefficient `0.002466`, |coef| `0.002466`

## Top 10 utility ridge features

- `lag_00__T1__molly`: coefficient `-0.002668` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `-0.002070` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.001934` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.001906` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.001764` (lowers CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.001692` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.001691` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001566` (raises CT win probability)
- `lag_01__T1__molly`: coefficient `-0.001300` (lowers CT win probability)
- `lag_11__T_active_smokes`: coefficient `-0.001188` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003789` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003530` (raises CT win probability)
- `lag_11__T1__duck_amount`: coefficient `0.003483` (raises CT win probability)
- `lag_03__CT_place_CONNECTOR`: coefficient `-0.003203` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003143` (raises CT win probability)
- `lag_07__T1__duck_amount`: coefficient `-0.003050` (lowers CT win probability)
- `lag_03__CT_place_MIDDLE`: coefficient `0.002971` (raises CT win probability)
- `lag_00__T1__alive`: coefficient `-0.002917` (lowers CT win probability)
- `lag_00__T1__hp`: coefficient `-0.002871` (lowers CT win probability)
- `lag_00__CT4__is_walking`: coefficient `-0.002820` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `90414`, seconds `84.50`, LSTM delta `+0.3991`

Top all feature movements:
- `lag_11__T1__duck_amount`: contribution `+0.013638`
- `lag_07__T1__duck_amount`: contribution `+0.011942`
- `lag_03__CT_place_CONNECTOR`: contribution `+0.011453`
- `lag_00__kill_diff_last_3s`: contribution `+0.009120`
- `lag_00__CT_kills_last_3s`: contribution `+0.009075`

Top utility-only movements:
- `lag_00__T1__molly`: contribution `+0.005907`

### tick `89582`, seconds `71.50`, LSTM delta `+0.3049`

Top all feature movements:
- `lag_09__T5__flash_duration`: contribution `+0.016483`
- `lag_11__T3__flash_duration`: contribution `+0.013867`
- `lag_07__CT_place_JUNGLE`: contribution `+0.011153`
- `lag_03__CT_place_STAIRS`: contribution `+0.010886`
- `lag_00__kill_diff_last_3s`: contribution `+0.009120`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `+0.016483`
- `lag_11__T3__flash_duration`: contribution `+0.013867`

### tick `86894`, seconds `29.50`, LSTM delta `-0.1937`

Top all feature movements:
- `lag_07__T4__flash_duration`: contribution `-0.010562`
- `lag_08__CT_place_SNIPERSNEST`: contribution `-0.009449`
- `lag_06__T3__is_scoped`: contribution `-0.009165`
- `lag_00__kill_diff_last_3s`: contribution `-0.009120`
- `lag_00__damage_diff_last_5s`: contribution `-0.007964`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `-0.010562`

### tick `88750`, seconds `58.50`, LSTM delta `-0.1810`

Top all feature movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.015982`
- `lag_02__T_flashes_last_5s`: contribution `-0.010061`
- `lag_14__CT_place_UNDERPASS`: contribution `-0.009238`
- `lag_00__kill_diff_last_3s`: contribution `-0.009120`
- `lag_10__CT_place_STAIRS`: contribution `-0.008750`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.015982`
- `lag_02__T_flashes_last_5s`: contribution `-0.010061`

### tick `89614`, seconds `72.00`, LSTM delta `-0.0844`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009120`
- `lag_00__CT_kills_last_3s`: contribution `-0.009075`
- `lag_00__CT4__is_walking`: contribution `+0.006726`
- `lag_15__T_place_UNDERPASS`: contribution `-0.006266`
- `lag_15__T4__flash_duration`: contribution `-0.004366`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `-0.004366`
