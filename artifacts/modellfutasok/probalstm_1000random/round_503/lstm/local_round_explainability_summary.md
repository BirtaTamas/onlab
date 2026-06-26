# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `22`

## Largest probability jumps

- tick `196915`, seconds `59.50`, LSTM `0.7526`, delta `+0.1349`
- tick `198451`, seconds `83.50`, LSTM `0.9002`, delta `+0.1129`
- tick `199059`, seconds `93.00`, LSTM `0.9400`, delta `+0.0949`
- tick `198483`, seconds `84.00`, LSTM `0.8234`, delta `-0.0769`
- tick `198675`, seconds `87.00`, LSTM `0.8346`, delta `+0.0450`
- tick `194387`, seconds `20.00`, LSTM `0.6486`, delta `+0.0359`
- tick `198547`, seconds `85.00`, LSTM `0.8217`, delta `-0.0344`
- tick `193363`, seconds `4.00`, LSTM `0.6616`, delta `-0.0336`
- tick `198515`, seconds `84.50`, LSTM `0.8561`, delta `+0.0327`
- tick `193523`, seconds `6.50`, LSTM `0.6302`, delta `-0.0301`

## Top 15 local ridge features

- `lag_14__T2__duck_amount`: coefficient `0.001462`, |coef| `0.001462`
- `lag_00__CT_kills_last_3s`: coefficient `0.001285`, |coef| `0.001285`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001258`, |coef| `0.001258`
- `lag_00__kill_diff_last_3s`: coefficient `0.001216`, |coef| `0.001216`
- `lag_05__CT_place_WALKWAY`: coefficient `-0.000976`, |coef| `0.000976`
- `lag_09__CT_place_CANAL`: coefficient `-0.000921`, |coef| `0.000921`
- `lag_00__CT_place_WATER`: coefficient `-0.000920`, |coef| `0.000920`
- `lag_05__CT_place_WATER`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_14__T1__is_walking`: coefficient `-0.000803`, |coef| `0.000803`
- `lag_00__damage_diff_last_5s`: coefficient `0.000750`, |coef| `0.000750`
- `lag_13__CT_place_WATER`: coefficient `0.000724`, |coef| `0.000724`
- `lag_00__CT_damage_last_5s`: coefficient `0.000724`, |coef| `0.000724`
- `lag_05__T_place_ALLEY`: coefficient `0.000724`, |coef| `0.000724`
- `lag_00__T_place_CANAL`: coefficient `-0.000724`, |coef| `0.000724`
- `lag_00__T5__alive`: coefficient `-0.000709`, |coef| `0.000709`

## Top 10 utility ridge features

- `lag_07__CT_B_site_active_infernos`: coefficient `0.000687` (raises CT win probability)
- `lag_03__CT5__smoke`: coefficient `-0.000551` (lowers CT win probability)
- `lag_07__CT_active_infernos`: coefficient `0.000467` (raises CT win probability)
- `lag_03__CT2__smoke`: coefficient `0.000438` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000423` (raises CT win probability)
- `lag_11__CT5__molly`: coefficient `-0.000389` (lowers CT win probability)
- `lag_02__CT4__smoke`: coefficient `-0.000382` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.000371` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000353` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000338` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T2__duck_amount`: coefficient `0.001462` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001285` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001258` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001216` (raises CT win probability)
- `lag_05__CT_place_WALKWAY`: coefficient `-0.000976` (lowers CT win probability)
- `lag_09__CT_place_CANAL`: coefficient `-0.000921` (lowers CT win probability)
- `lag_00__CT_place_WATER`: coefficient `-0.000920` (lowers CT win probability)
- `lag_05__CT_place_WATER`: coefficient `-0.000894` (lowers CT win probability)
- `lag_14__T1__is_walking`: coefficient `-0.000803` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000750` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `196915`, seconds `59.50`, LSTM delta `+0.1349`

Top all feature movements:
- `lag_14__T2__duck_amount`: contribution `+0.005591`
- `lag_05__CT_place_WATER`: contribution `+0.005433`
- `lag_05__CT_place_WALKWAY`: contribution `+0.004792`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004371`
- `lag_00__CT_kills_last_3s`: contribution `+0.003709`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002362`

### tick `198451`, seconds `83.50`, LSTM delta `+0.1129`

Top all feature movements:
- `lag_09__CT_place_CANAL`: contribution `+0.005595`
- `lag_14__T2__duck_amount`: contribution `+0.005591`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004371`
- `lag_14__CT_place_CANAL`: contribution `+0.003802`
- `lag_00__CT_kills_last_3s`: contribution `+0.003709`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `199059`, seconds `93.00`, LSTM delta `+0.0949`

Top all feature movements:
- `lag_13__CT_place_CONSTRUCTION`: contribution `+0.007692`
- `lag_06__CT_place_CONSTRUCTION`: contribution `+0.007308`
- `lag_00__CT_place_WATER`: contribution `+0.005592`
- `lag_00__CT_place_BACKOFA`: contribution `+0.005381`
- `lag_13__CT_place_WATER`: contribution `+0.004401`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.002780`
- `lag_08__T1__flash_duration`: contribution `+0.001975`

### tick `198483`, seconds `84.00`, LSTM delta `-0.0769`

Top all feature movements:
- `lag_14__T2__duck_amount`: contribution `-0.005591`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005246`
- `lag_00__kill_diff_last_3s`: contribution `-0.002926`
- `lag_12__CT1__duck_amount`: contribution `-0.002497`
- `lag_08__CT3__duck_amount`: contribution `-0.002111`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `198675`, seconds `87.00`, LSTM delta `+0.0450`

Top all feature movements:
- `lag_00__CT_place_WATER`: contribution `+0.005592`
- `lag_01__CT_place_CONSTRUCTION`: contribution `+0.004159`
- `lag_00__kill_diff_last_3s`: contribution `+0.002926`
- `lag_14__T1__is_walking`: contribution `+0.001833`
- `lag_01__CT_place_WATER`: contribution `-0.001757`

Top utility-only movements:
- No utility movement among the top local contributors.
