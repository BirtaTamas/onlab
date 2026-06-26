# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `21048`, seconds `31.00`, LSTM `0.9451`, delta `+0.0193`
- tick `19096`, seconds `0.50`, LSTM `0.9193`, delta `+0.0170`
- tick `19448`, seconds `6.00`, LSTM `0.9087`, delta `-0.0159`
- tick `21848`, seconds `43.50`, LSTM `0.9678`, delta `+0.0147`
- tick `20472`, seconds `22.00`, LSTM `0.9247`, delta `+0.0139`
- tick `19896`, seconds `13.00`, LSTM `0.9271`, delta `+0.0130`
- tick `21656`, seconds `40.50`, LSTM `0.9475`, delta `-0.0122`
- tick `20504`, seconds `22.50`, LSTM `0.9140`, delta `-0.0107`
- tick `19128`, seconds `1.00`, LSTM `0.9298`, delta `+0.0104`
- tick `19960`, seconds `14.00`, LSTM `0.9117`, delta `-0.0103`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000500`, |coef| `0.000500`
- `lag_00__kill_diff_last_3s`: coefficient `0.000416`, |coef| `0.000416`
- `lag_04__CT1__is_walking`: coefficient `-0.000406`, |coef| `0.000406`
- `lag_00__T2__alive`: coefficient `-0.000365`, |coef| `0.000365`
- `lag_00__CT_damage_last_5s`: coefficient `0.000359`, |coef| `0.000359`
- `lag_00__damage_diff_last_5s`: coefficient `0.000354`, |coef| `0.000354`
- `lag_00__CT_he_last_5s`: coefficient `0.000350`, |coef| `0.000350`
- `lag_00__T3__duck_amount`: coefficient `0.000348`, |coef| `0.000348`
- `lag_00__T3__is_walking`: coefficient `-0.000345`, |coef| `0.000345`
- `lag_10__T2__is_walking`: coefficient `-0.000340`, |coef| `0.000340`
- `lag_00__T2__hp`: coefficient `-0.000329`, |coef| `0.000329`
- `lag_00__CT5__is_walking`: coefficient `-0.000317`, |coef| `0.000317`
- `lag_00__T_walking_count`: coefficient `-0.000310`, |coef| `0.000310`
- `lag_00__CT1__is_walking`: coefficient `-0.000306`, |coef| `0.000306`
- `lag_06__T4__duck_amount`: coefficient `-0.000290`, |coef| `0.000290`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `0.000350` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.000230` (raises CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `-0.000222` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `0.000184` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000174` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000169` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000155` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000146` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000145` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000141` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000500` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000416` (raises CT win probability)
- `lag_04__CT1__is_walking`: coefficient `-0.000406` (lowers CT win probability)
- `lag_00__T2__alive`: coefficient `-0.000365` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000359` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000354` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `0.000348` (raises CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000345` (lowers CT win probability)
- `lag_10__T2__is_walking`: coefficient `-0.000340` (lowers CT win probability)
- `lag_00__T2__hp`: coefficient `-0.000329` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21048`, seconds `31.00`, LSTM delta `+0.0193`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.001442`
- `lag_00__kill_diff_last_3s`: contribution `+0.001002`
- `lag_04__CT1__is_walking`: contribution `+0.000948`
- `lag_13__T4__duck_amount`: contribution `+0.000874`
- `lag_00__T2__alive`: contribution `+0.000874`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19096`, seconds `0.50`, LSTM delta `+0.0170`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000900`
- `lag_01__T_place_TSPAWN`: contribution `+0.000822`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000750`
- `lag_01__centroid_distance_xy`: contribution `+0.000627`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000618`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000499`
- `lag_01__smoke_inv_diff`: contribution `+0.000469`
- `lag_01__molly_inv_diff`: contribution `+0.000445`
- `lag_01__CT_molly_inv`: contribution `+0.000341`
- `lag_01__CT_utility_inv`: contribution `+0.000235`

### tick `19448`, seconds `6.00`, LSTM delta `-0.0159`

Top all feature movements:
- `lag_00__CT_he_last_5s`: contribution `-0.006415`
- `lag_10__CT_he_last_5s`: contribution `-0.001373`
- `lag_02__CT_place_HELL`: contribution `-0.001295`
- `lag_06__T4__duck_amount`: contribution `-0.001072`
- `lag_00__CT_place_HELL`: contribution `-0.000746`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.006415`
- `lag_10__CT_he_last_5s`: contribution `-0.001373`
- `lag_01__CT2__molly`: contribution `-0.000237`

### tick `21848`, seconds `43.50`, LSTM delta `+0.0147`

Top all feature movements:
- `lag_11__CT_place_GARAGE`: contribution `+0.001962`
- `lag_00__CT_place_MINI`: contribution `+0.001747`
- `lag_00__CT_kills_last_3s`: contribution `+0.001442`
- `lag_00__kill_diff_last_3s`: contribution `+0.001002`
- `lag_04__CT1__is_walking`: contribution `+0.000948`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.000521`
- `lag_08__T3__flash_duration`: contribution `+0.000298`

### tick `20472`, seconds `22.00`, LSTM delta `+0.0139`

Top all feature movements:
- `lag_00__T3__duck_amount`: contribution `+0.001313`
- `lag_00__T3__is_walking`: contribution `+0.000800`
- `lag_10__T2__is_walking`: contribution `+0.000781`
- `lag_00__CT5__is_walking`: contribution `+0.000759`
- `lag_00__CT1__is_walking`: contribution `+0.000715`

Top utility-only movements:
- `lag_14__CT_B_site_active_infernos`: contribution `+0.000351`
