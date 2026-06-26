# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `110715`, seconds `7.00`, LSTM `0.9190`, delta `-0.0225`
- tick `113819`, seconds `55.50`, LSTM `0.9702`, delta `+0.0164`
- tick `110395`, seconds `2.00`, LSTM `0.9441`, delta `+0.0155`
- tick `111323`, seconds `16.50`, LSTM `0.9509`, delta `+0.0137`
- tick `110747`, seconds `7.50`, LSTM `0.9302`, delta `+0.0111`
- tick `111355`, seconds `17.00`, LSTM `0.9619`, delta `+0.0109`
- tick `111387`, seconds `17.50`, LSTM `0.9699`, delta `+0.0081`
- tick `113851`, seconds `56.00`, LSTM `0.9782`, delta `+0.0080`
- tick `110651`, seconds `6.00`, LSTM `0.9398`, delta `-0.0078`
- tick `111771`, seconds `23.50`, LSTM `0.9577`, delta `+0.0074`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000471`, |coef| `0.000471`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000290`, |coef| `0.000290`
- `lag_00__CT5__is_walking`: coefficient `-0.000278`, |coef| `0.000278`
- `lag_00__CT_place_TRUCK`: coefficient `0.000273`, |coef| `0.000273`
- `lag_00__CT_damage_last_5s`: coefficient `0.000270`, |coef| `0.000270`
- `lag_00__damage_diff_last_5s`: coefficient `0.000262`, |coef| `0.000262`
- `lag_00__CT_walking_count`: coefficient `-0.000261`, |coef| `0.000261`
- `lag_04__T3__duck_amount`: coefficient `-0.000225`, |coef| `0.000225`
- `lag_11__T_place_APARTMENTS`: coefficient `0.000221`, |coef| `0.000221`
- `lag_08__CT_place_STAIRS`: coefficient `0.000220`, |coef| `0.000220`
- `lag_00__CT3__is_walking`: coefficient `-0.000217`, |coef| `0.000217`
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000213`, |coef| `0.000213`
- `lag_10__CT_place_SNIPERSNEST`: coefficient `-0.000210`, |coef| `0.000210`
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000210`, |coef| `0.000210`
- `lag_00__T_place_HOUSE`: coefficient `-0.000200`, |coef| `0.000200`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000471` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000210` (lowers CT win probability)
- `lag_08__CT_smokes_last_5s`: coefficient `-0.000143` (lowers CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `-0.000133` (lowers CT win probability)
- `lag_13__CT_smokes_last_5s`: coefficient `-0.000107` (lowers CT win probability)
- `lag_11__CT_smokes_last_5s`: coefficient `0.000105` (raises CT win probability)
- `lag_05__CT_smokes_last_5s`: coefficient `0.000094` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.000094` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000072` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000069` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000290` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000278` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.000273` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000270` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000262` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000261` (lowers CT win probability)
- `lag_04__T3__duck_amount`: coefficient `-0.000225` (lowers CT win probability)
- `lag_11__T_place_APARTMENTS`: coefficient `0.000221` (raises CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `0.000220` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000217` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `110715`, seconds `7.00`, LSTM delta `-0.0225`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.008144`
- `lag_10__CT_smokes_last_5s`: contribution `-0.003624`
- `lag_00__T_place_HOUSE`: contribution `-0.000881`
- `lag_04__T3__duck_amount`: contribution `-0.000850`
- `lag_02__T_place_HOUSE`: contribution `-0.000720`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.008144`
- `lag_10__CT_smokes_last_5s`: contribution `-0.003624`

### tick `113819`, seconds `55.50`, LSTM delta `+0.0164`

Top all feature movements:
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.001126`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001007`
- `lag_04__T3__duck_amount`: contribution `+0.000850`
- `lag_00__CT5__is_walking`: contribution `+0.000666`
- `lag_00__damage_diff_last_5s`: contribution `+0.000592`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110395`, seconds `2.00`, LSTM delta `+0.0155`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.008144`
- `lag_00__T_place_SIDEALLEY`: contribution `+0.000678`
- `lag_02__T4__duck_amount`: contribution `+0.000479`
- `lag_03__T_velocity_mean`: contribution `+0.000469`
- `lag_04__T_place_TSPAWN`: contribution `+0.000391`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.008144`
- `lag_04__smoke_inv_diff`: contribution `+0.000082`

### tick `111323`, seconds `16.50`, LSTM delta `+0.0137`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `+0.001759`
- `lag_08__CT_place_STAIRS`: contribution `+0.001712`
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.000923`
- `lag_00__CT5__is_walking`: contribution `+0.000666`
- `lag_15__CT_place_SHOP`: contribution `+0.000612`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `+0.000331`

### tick `110747`, seconds `7.50`, LSTM delta `+0.0111`

Top all feature movements:
- `lag_11__CT_smokes_last_5s`: contribution `+0.001811`
- `lag_00__T_place_HOUSE`: contribution `+0.000881`
- `lag_04__T3__duck_amount`: contribution `+0.000850`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.000842`
- `lag_02__CT_place_SNIPERSNEST`: contribution `+0.000595`

Top utility-only movements:
- `lag_11__CT_smokes_last_5s`: contribution `+0.001811`
