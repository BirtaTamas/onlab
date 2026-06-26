# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `86776`, seconds `11.00`, LSTM `0.0999`, delta `-0.1470`
- tick `86648`, seconds `9.00`, LSTM `0.3738`, delta `-0.1013`
- tick `86680`, seconds `9.50`, LSTM `0.2772`, delta `-0.0966`
- tick `86872`, seconds `12.50`, LSTM `0.1446`, delta `+0.0867`
- tick `87000`, seconds `14.50`, LSTM `0.2417`, delta `+0.0810`
- tick `87192`, seconds `17.50`, LSTM `0.1193`, delta `-0.0612`
- tick `87256`, seconds `18.50`, LSTM `0.0230`, delta `-0.0576`
- tick `86936`, seconds `13.50`, LSTM `0.1220`, delta `-0.0482`
- tick `86968`, seconds `14.00`, LSTM `0.1608`, delta `+0.0388`
- tick `87224`, seconds `18.00`, LSTM `0.0806`, delta `-0.0387`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_00__kill_diff_last_3s`: coefficient `0.001092`, |coef| `0.001092`
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.000940`, |coef| `0.000940`
- `lag_00__damage_diff_last_5s`: coefficient `0.000940`, |coef| `0.000940`
- `lag_00__T_damage_last_5s`: coefficient `-0.000931`, |coef| `0.000931`
- `lag_05__T5__flash_duration`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000828`, |coef| `0.000828`
- `lag_00__CT3__utility_total`: coefficient `0.000824`, |coef| `0.000824`
- `lag_00__T_place_PALACEINTERIOR`: coefficient `-0.000765`, |coef| `0.000765`
- `lag_01__T_kills_last_3s`: coefficient `-0.000721`, |coef| `0.000721`
- `lag_13__T_place_PALACEINTERIOR`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_00__CT3__molly`: coefficient `0.000709`, |coef| `0.000709`
- `lag_00__CT4__alive`: coefficient `0.000704`, |coef| `0.000704`
- `lag_04__T_place_PALACEINTERIOR`: coefficient `-0.000702`, |coef| `0.000702`
- `lag_00__CT3__alive`: coefficient `0.000698`, |coef| `0.000698`

## Top 10 utility ridge features

- `lag_05__T5__flash_duration`: coefficient `-0.000909` (lowers CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000824` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000709` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `-0.000677` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.000661` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000637` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000616` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.000612` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000605` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.000592` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001294` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001092` (raises CT win probability)
- `lag_04__CT_place_SNIPERSNEST`: coefficient `-0.000940` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000940` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000931` (lowers CT win probability)
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000828` (raises CT win probability)
- `lag_00__T_place_PALACEINTERIOR`: coefficient `-0.000765` (lowers CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.000721` (lowers CT win probability)
- `lag_13__T_place_PALACEINTERIOR`: coefficient `-0.000717` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.000704` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `86776`, seconds `11.00`, LSTM delta `-0.1470`

Top all feature movements:
- `lag_05__CT_place_JUNGLE`: contribution `-0.004182`
- `lag_00__T_kills_last_3s`: contribution `-0.004100`
- `lag_08__T5__flash_duration`: contribution `-0.003251`
- `lag_08__CT_place_SNIPERSNEST`: contribution `-0.003141`
- `lag_07__CT_place_SHOP`: contribution `-0.002843`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.003251`
- `lag_08__T3__flash_duration`: contribution `-0.002350`
- `lag_00__T3__flash_duration`: contribution `-0.002230`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.002089`
- `lag_08__T_flash_duration_sum`: contribution `-0.001819`

### tick `86648`, seconds `9.00`, LSTM delta `-0.1013`

Top all feature movements:
- `lag_04__CT_place_SNIPERSNEST`: contribution `-0.005036`
- `lag_00__T_kills_last_3s`: contribution `-0.004100`
- `lag_03__CT_place_SHOP`: contribution `-0.003441`
- `lag_04__T5__flash_duration`: contribution `-0.002966`
- `lag_04__T3__flash_duration`: contribution `-0.002945`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.002966`
- `lag_04__T3__flash_duration`: contribution `-0.002945`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.002161`
- `lag_04__T_flash_duration_sum`: contribution `-0.001953`

### tick `86680`, seconds `9.50`, LSTM delta `-0.0966`

Top all feature movements:
- `lag_05__T5__flash_duration`: contribution `-0.004797`
- `lag_00__CT_place_JUNGLE`: contribution `-0.003555`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.003543`
- `lag_02__CT_place_JUNGLE`: contribution `-0.003539`
- `lag_09__CT_place_SHOP`: contribution `-0.003392`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.004797`
- `lag_05__T_flash_duration_sum`: contribution `-0.002193`
- `lag_05__T3__flash_duration`: contribution `-0.002107`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.001634`

### tick `86872`, seconds `12.50`, LSTM delta `+0.0867`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.005259`
- `lag_00__T_kills_last_3s`: contribution `+0.004100`
- `lag_06__CT_place_JUNGLE`: contribution `+0.003847`
- `lag_08__CT_place_JUNGLE`: contribution `+0.003586`
- `lag_10__CT_place_SHOP`: contribution `+0.003417`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.001882`
- `lag_02__T5__flash_duration`: contribution `+0.001723`
- `lag_03__T3__flash_duration`: contribution `+0.001665`
- `lag_11__T3__flash_duration`: contribution `+0.001308`

### tick `87000`, seconds `14.50`, LSTM delta `+0.0810`

Top all feature movements:
- `lag_10__CT_place_JUNGLE`: contribution `+0.004112`
- `lag_06__T5__flash_duration`: contribution `+0.003488`
- `lag_05__CT_place_STAIRS`: contribution `+0.003238`
- `lag_04__T_place_PALACEINTERIOR`: contribution `+0.002353`
- `lag_01__T_kills_last_3s`: contribution `+0.002284`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.003488`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.002161`
- `lag_07__T3__flash_duration`: contribution `+0.001648`
- `lag_15__T3__flash_duration`: contribution `+0.001517`
