# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `35219`, seconds `25.00`, LSTM `0.9382`, delta `+0.0489`
- tick `34899`, seconds `20.00`, LSTM `0.9244`, delta `+0.0420`
- tick `34099`, seconds `7.50`, LSTM `0.8312`, delta `-0.0270`
- tick `34579`, seconds `15.00`, LSTM `0.8788`, delta `+0.0258`
- tick `34515`, seconds `14.00`, LSTM `0.8415`, delta `-0.0250`
- tick `34163`, seconds `8.50`, LSTM `0.8579`, delta `+0.0244`
- tick `34195`, seconds `9.00`, LSTM `0.8813`, delta `+0.0234`
- tick `35123`, seconds `23.50`, LSTM `0.8825`, delta `-0.0209`
- tick `35379`, seconds `27.50`, LSTM `0.9365`, delta `+0.0175`
- tick `34675`, seconds `16.50`, LSTM `0.8651`, delta `-0.0173`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000670`, |coef| `0.000670`
- `lag_00__CT_place_HEAVEN`: coefficient `-0.000539`, |coef| `0.000539`
- `lag_00__CT_place_HUT`: coefficient `-0.000471`, |coef| `0.000471`
- `lag_01__CT_place_CONTROL`: coefficient `-0.000470`, |coef| `0.000470`
- `lag_06__CT_place_VENDING`: coefficient `-0.000454`, |coef| `0.000454`
- `lag_01__T4__is_walking`: coefficient `0.000442`, |coef| `0.000442`
- `lag_11__CT_place_TROPHY`: coefficient `0.000435`, |coef| `0.000435`
- `lag_12__CT_place_HUT`: coefficient `0.000432`, |coef| `0.000432`
- `lag_15__T_place_VENDING`: coefficient `0.000365`, |coef| `0.000365`
- `lag_00__CT3__is_walking`: coefficient `-0.000363`, |coef| `0.000363`
- `lag_00__T3__is_walking`: coefficient `-0.000349`, |coef| `0.000349`
- `lag_06__CT_place_TROPHY`: coefficient `0.000337`, |coef| `0.000337`
- `lag_00__CT_place_ADMIN`: coefficient `0.000333`, |coef| `0.000333`
- `lag_00__T_walking_count`: coefficient `-0.000332`, |coef| `0.000332`
- `lag_00__T_place_VENDING`: coefficient `-0.000330`, |coef| `0.000330`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `-0.000303` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.000190` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `-0.000189` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000187` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000179` (raises CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000173` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000166` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000163` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.000159` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `-0.000155` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000670` (raises CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `-0.000539` (lowers CT win probability)
- `lag_00__CT_place_HUT`: coefficient `-0.000471` (lowers CT win probability)
- `lag_01__CT_place_CONTROL`: coefficient `-0.000470` (lowers CT win probability)
- `lag_06__CT_place_VENDING`: coefficient `-0.000454` (lowers CT win probability)
- `lag_01__T4__is_walking`: coefficient `0.000442` (raises CT win probability)
- `lag_11__CT_place_TROPHY`: coefficient `0.000435` (raises CT win probability)
- `lag_12__CT_place_HUT`: coefficient `0.000432` (raises CT win probability)
- `lag_15__T_place_VENDING`: coefficient `0.000365` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000363` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `35219`, seconds `25.00`, LSTM delta `+0.0489`

Top all feature movements:
- `lag_06__CT_place_VENDING`: contribution `+0.007785`
- `lag_06__CT_place_TROPHY`: contribution `+0.004972`
- `lag_09__CT_place_VENDING`: contribution `+0.004788`
- `lag_03__CT_place_TROPHY`: contribution `+0.003946`
- `lag_00__CT_place_HEAVEN`: contribution `+0.002912`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `+0.001070`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.000615`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.000587`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.000547`
- `lag_02__CT_active_infernos`: contribution `+0.000438`

### tick `34899`, seconds `20.00`, LSTM delta `+0.0420`

Top all feature movements:
- `lag_11__CT_place_TROPHY`: contribution `+0.006430`
- `lag_12__CT_place_HUT`: contribution `+0.004210`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002326`
- `lag_10__CT_place_HEAVEN`: contribution `+0.001652`
- `lag_10__CT_place_HELL`: contribution `+0.001617`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `+0.000612`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.000559`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.000429`

### tick `34099`, seconds `7.50`, LSTM delta `-0.0270`

Top all feature movements:
- `lag_00__CT_place_HEAVEN`: contribution `-0.002912`
- `lag_01__CT_place_HEAVEN`: contribution `-0.001559`
- `lag_03__CT_place_ADMIN`: contribution `-0.001291`
- `lag_04__CT_place_HELL`: contribution `-0.001201`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.001070`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.001070`
- `lag_00__CT_active_infernos`: contribution `-0.000435`
- `lag_15__CT_molly_inv`: contribution `-0.000413`

### tick `34579`, seconds `15.00`, LSTM delta `+0.0258`

Top all feature movements:
- `lag_01__CT_place_CONTROL`: contribution `+0.004882`
- `lag_00__CT_place_HUT`: contribution `+0.004596`
- `lag_00__CT_place_HEAVEN`: contribution `-0.002912`
- `lag_01__CT_place_TROPHY`: contribution `+0.002278`
- `lag_15__CT_place_HELL`: contribution `+0.001461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34515`, seconds `14.00`, LSTM delta `-0.0250`

Top all feature movements:
- `lag_00__CT_place_HUT`: contribution `-0.004596`
- `lag_05__CT_place_CONTROL`: contribution `-0.001907`
- `lag_10__CT_place_HEAVEN`: contribution `-0.001652`
- `lag_10__CT_place_HELL`: contribution `-0.001617`
- `lag_12__T_place_ROOF`: contribution `-0.001448`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `-0.000659`
- `lag_02__CT_active_infernos`: contribution `-0.000438`
