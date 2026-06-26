# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `20265`, seconds `0.50`, LSTM `0.0157`, delta `-0.0370`
- tick `23145`, seconds `45.50`, LSTM `0.0110`, delta `-0.0357`
- tick `22825`, seconds `40.50`, LSTM `0.0295`, delta `-0.0160`
- tick `21897`, seconds `26.00`, LSTM `0.0632`, delta `-0.0147`
- tick `20841`, seconds `9.50`, LSTM `0.0591`, delta `+0.0133`
- tick `21929`, seconds `26.50`, LSTM `0.0512`, delta `-0.0120`
- tick `23113`, seconds `45.00`, LSTM `0.0466`, delta `+0.0116`
- tick `21129`, seconds `14.00`, LSTM `0.0514`, delta `-0.0104`
- tick `20713`, seconds `7.50`, LSTM `0.0361`, delta `+0.0097`
- tick `21577`, seconds `21.00`, LSTM `0.0752`, delta `+0.0096`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000597`, |coef| `0.000597`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000353`, |coef| `0.000353`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.000318`, |coef| `0.000318`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000296`, |coef| `0.000296`
- `lag_00__CT_velocity_mean`: coefficient `-0.000233`, |coef| `0.000233`
- `lag_00__T_velocity_mean`: coefficient `-0.000230`, |coef| `0.000230`
- `lag_00__T5__is_scoped`: coefficient `0.000210`, |coef| `0.000210`
- `lag_10__CT2__flash_duration`: coefficient `-0.000204`, |coef| `0.000204`
- `lag_11__T_place_LOWERTUNNEL`: coefficient `0.000203`, |coef| `0.000203`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000200`, |coef| `0.000200`
- `lag_10__T_place_MIDDOORS`: coefficient `-0.000199`, |coef| `0.000199`
- `lag_11__CT_place_HOLE`: coefficient `0.000192`, |coef| `0.000192`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000191`, |coef| `0.000191`
- `lag_07__T_place_TUNNELSTAIRS`: coefficient `0.000186`, |coef| `0.000186`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `0.000186`, |coef| `0.000186`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000597` (lowers CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.000204` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000185` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000171` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000164` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000138` (raises CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000131` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000131` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000115` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000114` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000353` (lowers CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.000318` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000296` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000233` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000230` (lowers CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.000210` (raises CT win probability)
- `lag_11__T_place_LOWERTUNNEL`: coefficient `0.000203` (raises CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000200` (lowers CT win probability)
- `lag_10__T_place_MIDDOORS`: coefficient `-0.000199` (lowers CT win probability)
- `lag_11__CT_place_HOLE`: coefficient `0.000192` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `20265`, seconds `0.50`, LSTM delta `-0.0370`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.005412`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001689`
- `lag_01__T_place_TSPAWN`: contribution `-0.001311`
- `lag_00__CT_velocity_mean`: contribution `-0.000801`
- `lag_00__T_velocity_mean`: contribution `-0.000800`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.005412`
- `lag_01__smoke_inv_diff`: contribution `-0.000470`
- `lag_01__utility_inv_diff`: contribution `-0.000397`
- `lag_01__molly_inv_diff`: contribution `-0.000301`
- `lag_01__T1__smoke`: contribution `-0.000195`

### tick `23145`, seconds `45.50`, LSTM delta `-0.0357`

Top all feature movements:
- `lag_11__CT_place_HOLE`: contribution `-0.002142`
- `lag_12__CT_place_HOLE`: contribution `-0.001893`
- `lag_10__CT2__flash_duration`: contribution `-0.001409`
- `lag_00__T_place_MIDDOORS`: contribution `-0.001351`
- `lag_10__CT_place_ARAMP`: contribution `-0.001056`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `-0.001409`

### tick `22825`, seconds `40.50`, LSTM delta `-0.0160`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `-0.001592`
- `lag_00__T_place_MIDDOORS`: contribution `-0.001351`
- `lag_02__CT_place_HOLE`: contribution `-0.001338`
- `lag_11__T_place_LOWERTUNNEL`: contribution `-0.000876`
- `lag_00__CT_place_ARAMP`: contribution `-0.000841`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.000354`
- `lag_03__T_active_infernos`: contribution `-0.000215`

### tick `21897`, seconds `26.00`, LSTM delta `-0.0147`

Top all feature movements:
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.001298`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `-0.001232`
- `lag_12__CT_place_SHORTSTAIRS`: contribution `-0.000704`
- `lag_00__CT5__duck_amount`: contribution `-0.000567`
- `lag_00__CT3__duck_amount`: contribution `-0.000511`

Top utility-only movements:
- `lag_02__T_active_infernos`: contribution `-0.000273`

### tick `20841`, seconds `9.50`, LSTM delta `+0.0133`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `+0.001338`
- `lag_07__CT_place_SHORTSTAIRS`: contribution `+0.000903`
- `lag_08__T_flashes_last_5s`: contribution `+0.000770`
- `lag_01__T_place_OUTSIDETUNNEL`: contribution `+0.000689`
- `lag_00__CT3__duck_amount`: contribution `+0.000628`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.000770`
- `lag_01__T1__smoke`: contribution `+0.000283`
