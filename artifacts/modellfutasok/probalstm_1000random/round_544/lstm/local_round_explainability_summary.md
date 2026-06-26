# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `20`

## Largest probability jumps

- tick `172425`, seconds `103.00`, LSTM `0.1152`, delta `-0.4164`
- tick `172105`, seconds `98.00`, LSTM `0.5599`, delta `-0.2027`
- tick `171369`, seconds `86.50`, LSTM `0.7169`, delta `+0.1119`
- tick `172521`, seconds `104.50`, LSTM `0.0147`, delta `-0.0599`
- tick `172457`, seconds `103.50`, LSTM `0.0725`, delta `-0.0427`
- tick `172137`, seconds `98.50`, LSTM `0.5243`, delta `-0.0356`
- tick `171497`, seconds `88.50`, LSTM `0.7464`, delta `+0.0342`
- tick `172393`, seconds `102.50`, LSTM `0.5316`, delta `-0.0338`
- tick `171977`, seconds `96.00`, LSTM `0.7462`, delta `-0.0318`
- tick `172233`, seconds `100.00`, LSTM `0.5458`, delta `+0.0307`

## Top 15 local ridge features

- `lag_14__T_place_QUAD`: coefficient `0.004572`, |coef| `0.004572`
- `lag_08__T_place_QUAD`: coefficient `-0.002833`, |coef| `0.002833`
- `lag_00__T_place_BALCONY`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_01__T_place_BALCONY`: coefficient `-0.002226`, |coef| `0.002226`
- `lag_04__T_place_QUAD`: coefficient `0.001892`, |coef| `0.001892`
- `lag_09__CT1__flash_duration`: coefficient `0.001877`, |coef| `0.001877`
- `lag_00__T_place_QUAD`: coefficient `0.001820`, |coef| `0.001820`
- `lag_06__CT4__flash_duration`: coefficient `-0.001780`, |coef| `0.001780`
- `lag_06__T2__flash_duration`: coefficient `-0.001742`, |coef| `0.001742`
- `lag_00__kill_diff_last_3s`: coefficient `0.001631`, |coef| `0.001631`
- `lag_06__CT3__flash_duration`: coefficient `-0.001629`, |coef| `0.001629`
- `lag_00__CT4__flash_duration`: coefficient `0.001549`, |coef| `0.001549`
- `lag_00__CT_place_PIT`: coefficient `0.001523`, |coef| `0.001523`
- `lag_01__T_bomb_zone_count`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_06__CT_flash_duration_sum`: coefficient `-0.001457`, |coef| `0.001457`

## Top 10 utility ridge features

- `lag_09__CT1__flash_duration`: coefficient `0.001877` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.001780` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.001742` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.001629` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001549` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.001457` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.001142` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.001068` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.001054` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.001006` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_QUAD`: coefficient `0.004572` (raises CT win probability)
- `lag_08__T_place_QUAD`: coefficient `-0.002833` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.002249` (lowers CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `-0.002226` (lowers CT win probability)
- `lag_04__T_place_QUAD`: coefficient `0.001892` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.001820` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001631` (raises CT win probability)
- `lag_00__CT_place_PIT`: coefficient `0.001523` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.001500` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001420` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `172425`, seconds `103.00`, LSTM delta `-0.4164`

Top all feature movements:
- `lag_14__T_place_QUAD`: contribution `-0.110124`
- `lag_00__T_place_BALCONY`: contribution `-0.030924`
- `lag_01__T_place_BALCONY`: contribution `-0.030604`
- `lag_06__CT4__flash_duration`: contribution `-0.011753`
- `lag_06__T2__flash_duration`: contribution `-0.010251`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `-0.011753`
- `lag_06__T2__flash_duration`: contribution `-0.010251`
- `lag_00__CT4__flash_duration`: contribution `-0.010225`
- `lag_06__CT3__flash_duration`: contribution `-0.008251`
- `lag_06__CT_flash_duration_sum`: contribution `-0.007602`

### tick `172105`, seconds `98.00`, LSTM delta `-0.2027`

Top all feature movements:
- `lag_08__T_place_QUAD`: contribution `-0.068245`
- `lag_04__T_place_QUAD`: contribution `-0.045577`
- `lag_05__CT_place_LIBRARY`: contribution `-0.005098`
- `lag_06__CT3__is_scoped`: contribution `-0.004371`
- `lag_00__T_kills_last_3s`: contribution `-0.004007`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `-0.003151`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.002577`
- `lag_02__T5__flash_duration`: contribution `-0.001741`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.001706`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.001659`

### tick `171369`, seconds `86.50`, LSTM delta `+0.1119`

Top all feature movements:
- `lag_09__CT1__flash_duration`: contribution `+0.014050`
- `lag_09__CT2__flash_duration`: contribution `+0.006963`
- `lag_09__CT_flash_duration_sum`: contribution `+0.006552`
- `lag_00__kill_diff_last_3s`: contribution `+0.003926`
- `lag_08__T5__is_scoped`: contribution `+0.003918`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `+0.014050`
- `lag_09__CT2__flash_duration`: contribution `+0.006963`
- `lag_09__CT_flash_duration_sum`: contribution `+0.006552`

### tick `172521`, seconds `104.50`, LSTM delta `-0.0599`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `-0.016666`
- `lag_03__T_place_BALCONY`: contribution `-0.014700`
- `lag_01__T_bomb_zone_count`: contribution `+0.008732`
- `lag_09__CT_flash_duration_sum`: contribution `+0.005576`
- `lag_06__T5__is_scoped`: contribution `+0.004056`

Top utility-only movements:
- `lag_09__CT_flash_duration_sum`: contribution `+0.005576`
- `lag_09__CT4__flash_duration`: contribution `-0.002239`
- `lag_09__T2__flash_duration`: contribution `-0.002072`

### tick `172457`, seconds `103.50`, LSTM delta `-0.0427`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `-0.030604`
- `lag_00__T_shots_fired_sum`: contribution `+0.008520`
- `lag_02__T_place_BALCONY`: contribution `-0.005292`
- `lag_08__T5__is_scoped`: contribution `-0.003918`
- `lag_15__T_place_QUAD`: contribution `+0.003616`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `-0.001267`
