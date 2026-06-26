# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `18134`, seconds `67.00`, LSTM `0.9462`, delta `+0.0281`
- tick `18166`, seconds `67.50`, LSTM `0.9694`, delta `+0.0232`
- tick `14678`, seconds `13.00`, LSTM `0.9053`, delta `+0.0232`
- tick `13878`, seconds `0.50`, LSTM `0.9166`, delta `+0.0211`
- tick `17814`, seconds `62.00`, LSTM `0.9139`, delta `+0.0180`
- tick `14422`, seconds `9.00`, LSTM `0.9222`, delta `-0.0176`
- tick `14870`, seconds `16.00`, LSTM `0.9297`, delta `+0.0173`
- tick `14550`, seconds `11.00`, LSTM `0.9080`, delta `-0.0160`
- tick `14646`, seconds `12.50`, LSTM `0.8821`, delta `-0.0114`
- tick `14582`, seconds `11.50`, LSTM `0.8976`, delta `-0.0103`

## Top 15 local ridge features

- `lag_10__T_place_CONTROL`: coefficient `0.000438`, |coef| `0.000438`
- `lag_12__CT_place_HELL`: coefficient `-0.000437`, |coef| `0.000437`
- `lag_13__CT_place_HELL`: coefficient `-0.000325`, |coef| `0.000325`
- `lag_00__CT_place_ADMIN`: coefficient `0.000322`, |coef| `0.000322`
- `lag_00__T_place_CONTROL`: coefficient `-0.000312`, |coef| `0.000312`
- `lag_00__CT3__is_walking`: coefficient `-0.000299`, |coef| `0.000299`
- `lag_09__T_place_CONTROL`: coefficient `0.000290`, |coef| `0.000290`
- `lag_07__T_place_TROPHY`: coefficient `0.000285`, |coef| `0.000285`
- `lag_12__CT_place_ADMIN`: coefficient `0.000279`, |coef| `0.000279`
- `lag_06__CT3__duck_amount`: coefficient `0.000279`, |coef| `0.000279`
- `lag_08__T_place_TROPHY`: coefficient `0.000276`, |coef| `0.000276`
- `lag_03__T_place_TROPHY`: coefficient `-0.000270`, |coef| `0.000270`
- `lag_08__T_place_VENDING`: coefficient `-0.000268`, |coef| `0.000268`
- `lag_00__T_place_VENDING`: coefficient `-0.000248`, |coef| `0.000248`
- `lag_11__T_place_CONTROL`: coefficient `0.000245`, |coef| `0.000245`

## Top 10 utility ridge features

- `lag_13__CT_flashes_last_5s`: coefficient `0.000231` (raises CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000157` (raises CT win probability)
- `lag_07__CT_flashes_last_5s`: coefficient `0.000156` (raises CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `-0.000153` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `-0.000136` (lowers CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `-0.000133` (lowers CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `0.000117` (raises CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.000107` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000103` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000101` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_CONTROL`: coefficient `0.000438` (raises CT win probability)
- `lag_12__CT_place_HELL`: coefficient `-0.000437` (lowers CT win probability)
- `lag_13__CT_place_HELL`: coefficient `-0.000325` (lowers CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `0.000322` (raises CT win probability)
- `lag_00__T_place_CONTROL`: coefficient `-0.000312` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000299` (lowers CT win probability)
- `lag_09__T_place_CONTROL`: coefficient `0.000290` (raises CT win probability)
- `lag_07__T_place_TROPHY`: coefficient `0.000285` (raises CT win probability)
- `lag_12__CT_place_ADMIN`: coefficient `0.000279` (raises CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `0.000279` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18134`, seconds `67.00`, LSTM delta `+0.0281`

Top all feature movements:
- `lag_10__T_place_CONTROL`: contribution `+0.003115`
- `lag_00__T_place_CONTROL`: contribution `+0.002219`
- `lag_09__T_place_CONTROL`: contribution `+0.002061`
- `lag_07__T_place_TROPHY`: contribution `+0.001807`
- `lag_03__T_place_TROPHY`: contribution `+0.001713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18166`, seconds `67.50`, LSTM delta `+0.0232`

Top all feature movements:
- `lag_10__T_place_CONTROL`: contribution `+0.003115`
- `lag_00__T_place_CONTROL`: contribution `+0.002219`
- `lag_08__T_place_TROPHY`: contribution `+0.001751`
- `lag_11__T_place_CONTROL`: contribution `+0.001738`
- `lag_03__T_place_TROPHY`: contribution `+0.001713`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14678`, seconds `13.00`, LSTM delta `+0.0232`

Top all feature movements:
- `lag_12__CT_place_HELL`: contribution `+0.004740`
- `lag_12__CT_place_ADMIN`: contribution `+0.001937`
- `lag_13__CT_place_HELL`: contribution `+0.001765`
- `lag_08__CT_place_ADMIN`: contribution `+0.001362`
- `lag_00__CT_place_HEAVEN`: contribution `+0.001210`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13878`, seconds `0.50`, LSTM delta `+0.0211`

Top all feature movements:
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000687`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000663`
- `lag_01__T_place_TSPAWN`: contribution `+0.000644`
- `lag_01__centroid_distance_xy`: contribution `+0.000612`
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000597`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000197`
- `lag_01__CT4__molly`: contribution `+0.000164`
- `lag_01__CT4__utility_total`: contribution `+0.000162`
- `lag_00__CT1__flash`: contribution `+0.000158`
- `lag_01__smoke_inv_diff`: contribution `+0.000157`

### tick `17814`, seconds `62.00`, LSTM delta `+0.0180`

Top all feature movements:
- `lag_12__CT_place_HELL`: contribution `+0.002370`
- `lag_00__T_place_CONTROL`: contribution `-0.002219`
- `lag_12__CT_place_ADMIN`: contribution `+0.001937`
- `lag_07__T_place_TROPHY`: contribution `+0.001807`
- `lag_07__T_place_VENDING`: contribution `+0.001186`

Top utility-only movements:
- No utility movement among the top local contributors.
