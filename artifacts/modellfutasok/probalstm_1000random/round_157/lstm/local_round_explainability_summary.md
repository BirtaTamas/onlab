# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `5`

## Largest probability jumps

- tick `49687`, seconds `92.50`, LSTM `0.0548`, delta `-0.3160`
- tick `46775`, seconds `47.00`, LSTM `0.4066`, delta `-0.2491`
- tick `49399`, seconds `88.00`, LSTM `0.5689`, delta `+0.1655`
- tick `49463`, seconds `89.00`, LSTM `0.4078`, delta `-0.1520`
- tick `46967`, seconds `50.00`, LSTM `0.4830`, delta `+0.0867`
- tick `44535`, seconds `12.00`, LSTM `0.5953`, delta `+0.0741`
- tick `49591`, seconds `91.00`, LSTM `0.2996`, delta `-0.0738`
- tick `46743`, seconds `46.50`, LSTM `0.6557`, delta `+0.0588`
- tick `44439`, seconds `10.50`, LSTM `0.4982`, delta `+0.0528`
- tick `45175`, seconds `22.00`, LSTM `0.5464`, delta `+0.0433`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002717`, |coef| `0.002717`
- `lag_14__T_flashes_last_5s`: coefficient `0.002372`, |coef| `0.002372`
- `lag_00__T_kills_last_3s`: coefficient `-0.002363`, |coef| `0.002363`
- `lag_15__T_flashes_last_5s`: coefficient `0.002274`, |coef| `0.002274`
- `lag_09__CT_place_FOUNTAIN`: coefficient `0.001867`, |coef| `0.001867`
- `lag_05__T_flashes_last_5s`: coefficient `-0.001740`, |coef| `0.001740`
- `lag_00__damage_diff_last_5s`: coefficient `0.001675`, |coef| `0.001675`
- `lag_02__T_place_MAIN`: coefficient `0.001670`, |coef| `0.001670`
- `lag_13__CT_place_HEAVEN`: coefficient `0.001639`, |coef| `0.001639`
- `lag_12__T_shots_fired_sum`: coefficient `0.001569`, |coef| `0.001569`
- `lag_00__T_damage_last_5s`: coefficient `-0.001541`, |coef| `0.001541`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001533`, |coef| `0.001533`
- `lag_03__CT_place_BACKOFB`: coefficient `0.001488`, |coef| `0.001488`
- `lag_02__CT_place_CANAL`: coefficient `-0.001461`, |coef| `0.001461`
- `lag_05__T_place_CANAL`: coefficient `0.001410`, |coef| `0.001410`

## Top 10 utility ridge features

- `lag_14__T_flashes_last_5s`: coefficient `0.002372` (raises CT win probability)
- `lag_15__T_flashes_last_5s`: coefficient `0.002274` (raises CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.001740` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.001391` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001359` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.001318` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000960` (lowers CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000914` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.000824` (raises CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `0.000793` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002717` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002363` (lowers CT win probability)
- `lag_09__CT_place_FOUNTAIN`: coefficient `0.001867` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001675` (raises CT win probability)
- `lag_02__T_place_MAIN`: coefficient `0.001670` (raises CT win probability)
- `lag_13__CT_place_HEAVEN`: coefficient `0.001639` (raises CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `0.001569` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001541` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001533` (raises CT win probability)
- `lag_03__CT_place_BACKOFB`: coefficient `0.001488` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `49687`, seconds `92.50`, LSTM delta `-0.3160`

Top all feature movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.021495`
- `lag_03__CT_place_BRICKS`: contribution `-0.018997`
- `lag_01__CT_place_BRICKS`: contribution `-0.011231`
- `lag_02__T_place_MAIN`: contribution `-0.010794`
- `lag_01__T_place_MAIN`: contribution `-0.009014`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.021495`
- `lag_08__CT2__flash_duration`: contribution `-0.007634`
- `lag_07__CT2__flash_duration`: contribution `-0.004240`

### tick `46775`, seconds `47.00`, LSTM delta `-0.2491`

Top all feature movements:
- `lag_09__CT_place_FOUNTAIN`: contribution `-0.019635`
- `lag_05__T3__flash_duration`: contribution `-0.009089`
- `lag_13__CT_place_HEAVEN`: contribution `-0.008850`
- `lag_11__T_place_MAIN`: contribution `-0.008656`
- `lag_00__T_kills_last_3s`: contribution `-0.007485`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.009089`
- `lag_03__CT4__flash_duration`: contribution `-0.006592`

### tick `49399`, seconds `88.00`, LSTM delta `+0.1655`

Top all feature movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.020603`
- `lag_05__T_flashes_last_5s`: contribution `+0.015768`
- `lag_10__T_place_MAIN`: contribution `+0.006616`
- `lag_00__kill_diff_last_3s`: contribution `+0.006540`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005327`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.020603`
- `lag_05__T_flashes_last_5s`: contribution `+0.015768`
- `lag_15__T2__flash`: contribution `+0.001906`

### tick `49463`, seconds `89.00`, LSTM delta `-0.1520`

Top all feature movements:
- `lag_02__T_place_MAIN`: contribution `-0.010794`
- `lag_00__T_kills_last_3s`: contribution `-0.007485`
- `lag_00__T_shots_fired_sum`: contribution `-0.007309`
- `lag_07__T_flashes_last_5s`: contribution `-0.007182`
- `lag_14__T_place_MAIN`: contribution `-0.006853`

Top utility-only movements:
- `lag_07__T_flashes_last_5s`: contribution `-0.007182`
- `lag_01__CT2__flash_duration`: contribution `-0.005293`
- `lag_00__CT2__flash_duration`: contribution `-0.004383`
- `lag_01__CT_flash_duration_sum`: contribution `-0.001654`

### tick `46967`, seconds `50.00`, LSTM delta `+0.0867`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `+0.007485`
- `lag_00__kill_diff_last_3s`: contribution `+0.006540`
- `lag_15__CT_place_FOUNTAIN`: contribution `+0.004168`
- `lag_00__T2__duck_amount`: contribution `+0.004016`
- `lag_06__T_shots_fired_sum`: contribution `+0.003611`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.002741`
- `lag_09__CT4__flash_duration`: contribution `+0.002737`
- `lag_11__T3__flash_duration`: contribution `+0.002721`
