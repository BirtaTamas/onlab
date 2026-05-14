# Local Round Explainability

- csv_path: `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `9968`, seconds `1.50`, LSTM `0.9254`, delta `+0.0334`
- tick `9904`, seconds `0.50`, LSTM `0.9007`, delta `+0.0189`
- tick `13456`, seconds `56.00`, LSTM `0.9626`, delta `+0.0186`
- tick `13584`, seconds `58.00`, LSTM `0.9752`, delta `+0.0139`
- tick `12656`, seconds `43.50`, LSTM `0.9268`, delta `-0.0132`
- tick `12016`, seconds `33.50`, LSTM `0.9296`, delta `-0.0130`
- tick `10576`, seconds `11.00`, LSTM `0.9337`, delta `-0.0109`
- tick `12144`, seconds `35.50`, LSTM `0.9485`, delta `+0.0109`
- tick `13232`, seconds `52.50`, LSTM `0.9407`, delta `+0.0105`
- tick `12432`, seconds `40.00`, LSTM `0.9339`, delta `-0.0099`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.000640`, |coef| `0.000640`
- `lag_00__CT3__is_walking`: coefficient `-0.000579`, |coef| `0.000579`
- `lag_00__CT_walking_count`: coefficient `-0.000323`, |coef| `0.000323`
- `lag_00__T_walking_count`: coefficient `-0.000304`, |coef| `0.000304`
- `lag_00__T3__is_walking`: coefficient `-0.000300`, |coef| `0.000300`
- `lag_00__CT_kills_last_3s`: coefficient `0.000298`, |coef| `0.000298`
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.000296`, |coef| `0.000296`
- `lag_00__CT_damage_last_5s`: coefficient `0.000239`, |coef| `0.000239`
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.000236`, |coef| `0.000236`
- `lag_06__CT2__duck_amount`: coefficient `-0.000234`, |coef| `0.000234`
- `lag_00__kill_diff_last_3s`: coefficient `0.000231`, |coef| `0.000231`
- `lag_00__T4__is_walking`: coefficient `-0.000230`, |coef| `0.000230`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000223`, |coef| `0.000223`
- `lag_15__CT1__duck_amount`: coefficient `-0.000217`, |coef| `0.000217`
- `lag_00__damage_diff_last_5s`: coefficient `0.000196`, |coef| `0.000196`

## Top 10 utility ridge features

- `lag_12__CT_active_smokes`: coefficient `-0.000124` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000114` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `-0.000109` (lowers CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `-0.000105` (lowers CT win probability)
- `lag_03__CT_B_site_active_smokes`: coefficient `-0.000098` (lowers CT win probability)
- `lag_00__CT_active_infernos`: coefficient `-0.000095` (lowers CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `-0.000094` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `0.000092` (raises CT win probability)
- `lag_06__CT_B_site_active_smokes`: coefficient `-0.000091` (lowers CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000090` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.000640` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000579` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000323` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000304` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000300` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000298` (raises CT win probability)
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.000296` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000239` (raises CT win probability)
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.000236` (raises CT win probability)
- `lag_06__CT2__duck_amount`: coefficient `-0.000234` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9968`, seconds `1.50`, LSTM delta `+0.0334`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `+0.017400`
- `lag_03__CT_place_TSIDELOWER`: contribution `+0.008014`
- `lag_02__CT_velocity_mean`: contribution `+0.000327`
- `lag_03__T_place_TSPAWN`: contribution `+0.000309`
- `lag_03__armor_diff`: contribution `+0.000212`

Top utility-only movements:
- `lag_00__CT2__smoke`: contribution `+0.000132`
- `lag_03__molly_inv_diff`: contribution `+0.000112`
- `lag_03__utility_inv_diff`: contribution `+0.000080`

### tick `9904`, seconds `0.50`, LSTM delta `+0.0189`

Top all feature movements:
- `lag_01__CT_place_TSIDELOWER`: contribution `+0.006402`
- `lag_00__CT_velocity_mean`: contribution `+0.000455`
- `lag_01__T_place_TSPAWN`: contribution `+0.000375`
- `lag_01__armor_diff`: contribution `+0.000234`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000175`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `+0.000150`
- `lag_01__utility_inv_diff`: contribution `+0.000116`
- `lag_01__CT3__utility_total`: contribution `+0.000113`
- `lag_01__CT3__smoke`: contribution `+0.000087`

### tick `13456`, seconds `56.00`, LSTM delta `+0.0186`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `+0.001202`
- `lag_00__CT_kills_last_3s`: contribution `+0.000859`
- `lag_13__CT_shots_fired_sum`: contribution `+0.000673`
- `lag_12__CT2__shots_fired`: contribution `+0.000629`
- `lag_00__kill_diff_last_3s`: contribution `+0.000557`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13584`, seconds `58.00`, LSTM delta `+0.0139`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.000859`
- `lag_00__CT2__duck_amount`: contribution `-0.000570`
- `lag_00__kill_diff_last_3s`: contribution `+0.000557`
- `lag_00__CT5__duck_amount`: contribution `+0.000523`
- `lag_00__CT_damage_last_5s`: contribution `+0.000520`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `12656`, seconds `43.50`, LSTM delta `-0.0132`

Top all feature movements:
- `lag_00__CT3__is_walking`: contribution `-0.001383`
- `lag_15__CT1__duck_amount`: contribution `-0.000816`
- `lag_03__CT4__duck_amount`: contribution `-0.000461`
- `lag_12__CT4__is_walking`: contribution `-0.000431`
- `lag_10__CT3__duck_amount`: contribution `-0.000417`

Top utility-only movements:
- No utility movement among the top local contributors.
