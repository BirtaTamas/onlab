# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `148645`, seconds `94.50`, LSTM `0.9154`, delta `-0.0569`
- tick `147717`, seconds `80.00`, LSTM `0.8924`, delta `+0.0407`
- tick `144293`, seconds `26.50`, LSTM `0.7785`, delta `-0.0405`
- tick `144261`, seconds `26.00`, LSTM `0.8190`, delta `+0.0326`
- tick `142629`, seconds `0.50`, LSTM `0.8106`, delta `-0.0296`
- tick `142661`, seconds `1.00`, LSTM `0.7840`, delta `-0.0266`
- tick `147749`, seconds `80.50`, LSTM `0.9182`, delta `+0.0258`
- tick `144805`, seconds `34.50`, LSTM `0.7705`, delta `+0.0257`
- tick `142757`, seconds `2.50`, LSTM `0.8294`, delta `+0.0235`
- tick `143365`, seconds `12.00`, LSTM `0.8216`, delta `+0.0228`

## Top 15 local ridge features

- `lag_00__CT_walking_count`: coefficient `-0.000991`, |coef| `0.000991`
- `lag_00__T_place_IVY`: coefficient `-0.000960`, |coef| `0.000960`
- `lag_12__CT_place_TUNNELS`: coefficient `0.000896`, |coef| `0.000896`
- `lag_11__CT_place_TUNNELS`: coefficient `0.000844`, |coef| `0.000844`
- `lag_00__CT5__is_walking`: coefficient `-0.000837`, |coef| `0.000837`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000798`, |coef| `0.000798`
- `lag_00__CT_place_BACKOFB`: coefficient `-0.000792`, |coef| `0.000792`
- `lag_05__T2__duck_amount`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_01__CT5__is_walking`: coefficient `0.000755`, |coef| `0.000755`
- `lag_00__CT2__is_walking`: coefficient `-0.000710`, |coef| `0.000710`
- `lag_07__CT4__is_walking`: coefficient `0.000703`, |coef| `0.000703`
- `lag_04__CT4__duck_amount`: coefficient `-0.000690`, |coef| `0.000690`
- `lag_00__CT3__is_walking`: coefficient `-0.000677`, |coef| `0.000677`
- `lag_00__damage_diff_last_5s`: coefficient `0.000672`, |coef| `0.000672`
- `lag_02__T_place_TSTAIRS`: coefficient `-0.000659`, |coef| `0.000659`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_smokes`: coefficient `-0.000391` (lowers CT win probability)
- `lag_15__CT_A_site_active_smokes`: coefficient `-0.000306` (lowers CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.000280` (lowers CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `-0.000269` (lowers CT win probability)
- `lag_13__CT_A_site_active_smokes`: coefficient `-0.000250` (lowers CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `-0.000246` (lowers CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `-0.000242` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `-0.000235` (lowers CT win probability)
- `lag_01__CT4__utility_total`: coefficient `-0.000233` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.000231` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_walking_count`: coefficient `-0.000991` (lowers CT win probability)
- `lag_00__T_place_IVY`: coefficient `-0.000960` (lowers CT win probability)
- `lag_12__CT_place_TUNNELS`: coefficient `0.000896` (raises CT win probability)
- `lag_11__CT_place_TUNNELS`: coefficient `0.000844` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000837` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000798` (raises CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `-0.000792` (lowers CT win probability)
- `lag_05__T2__duck_amount`: coefficient `-0.000776` (lowers CT win probability)
- `lag_01__CT5__is_walking`: coefficient `0.000755` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.000710` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `148645`, seconds `94.50`, LSTM delta `-0.0569`

Top all feature movements:
- `lag_12__CT_place_TUNNELS`: contribution `-0.002742`
- `lag_10__T_duck_amount_mean`: contribution `-0.002139`
- `lag_03__T_duck_amount_mean`: contribution `-0.001981`
- `lag_08__T_duck_amount_mean`: contribution `-0.001972`
- `lag_12__T_duck_amount_mean`: contribution `-0.001807`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `147717`, seconds `80.00`, LSTM delta `+0.0407`

Top all feature movements:
- `lag_05__T2__duck_amount`: contribution `+0.002965`
- `lag_11__CT_place_TUNNELS`: contribution `+0.002584`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001664`
- `lag_00__CT2__duck_amount`: contribution `+0.001379`
- `lag_00__T3__is_walking`: contribution `+0.001355`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144293`, seconds `26.50`, LSTM delta `-0.0405`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.004992`
- `lag_15__CT_place_BACKOFB`: contribution `-0.002651`
- `lag_04__CT4__duck_amount`: contribution `+0.002534`
- `lag_00__CT5__is_walking`: contribution `-0.002007`
- `lag_01__CT5__is_walking`: contribution `-0.001809`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144261`, seconds `26.00`, LSTM delta `+0.0326`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004437`
- `lag_00__CT5__is_walking`: contribution `+0.002007`
- `lag_07__CT4__is_walking`: contribution `+0.001675`
- `lag_00__CT5__duck_amount`: contribution `+0.001551`
- `lag_00__CT1__shots_fired`: contribution `+0.001538`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `142629`, seconds `0.50`, LSTM delta `-0.0296`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001896`
- `lag_00__T2__armor`: contribution `-0.001422`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001374`
- `lag_01__T_place_TSPAWN`: contribution `-0.001230`
- `lag_01__centroid_distance_xy`: contribution `-0.001230`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000728`
- `lag_01__CT4__utility_total`: contribution `-0.000668`
- `lag_01__CT4__flash`: contribution `-0.000592`
- `lag_01__molly_inv_diff`: contribution `-0.000522`
- `lag_01__smoke_inv_diff`: contribution `-0.000518`
