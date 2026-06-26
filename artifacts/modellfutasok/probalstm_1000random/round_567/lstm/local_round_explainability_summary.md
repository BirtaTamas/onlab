# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `11`

## Largest probability jumps

- tick `103212`, seconds `0.50`, LSTM `0.0223`, delta `-0.0348`
- tick `105196`, seconds `31.50`, LSTM `0.0390`, delta `-0.0335`
- tick `105164`, seconds `31.00`, LSTM `0.0725`, delta `-0.0241`
- tick `105100`, seconds `30.00`, LSTM `0.0887`, delta `+0.0133`
- tick `105004`, seconds `28.50`, LSTM `0.0673`, delta `-0.0128`
- tick `105580`, seconds `37.50`, LSTM `0.0084`, delta `-0.0116`
- tick `104076`, seconds `14.00`, LSTM `0.0548`, delta `+0.0101`
- tick `105228`, seconds `32.00`, LSTM `0.0295`, delta `-0.0095`
- tick `104396`, seconds `19.00`, LSTM `0.0574`, delta `+0.0083`
- tick `103948`, seconds `12.00`, LSTM `0.0383`, delta `+0.0081`

## Top 15 local ridge features

- `lag_06__T_place_IVY`: coefficient `0.000497`, |coef| `0.000497`
- `lag_14__CT1__flash_duration`: coefficient `0.000422`, |coef| `0.000422`
- `lag_05__T_place_IVY`: coefficient `0.000416`, |coef| `0.000416`
- `lag_06__T_place_TUNNELS`: coefficient `-0.000385`, |coef| `0.000385`
- `lag_13__CT1__flash_duration`: coefficient `0.000342`, |coef| `0.000342`
- `lag_05__T4__is_scoped`: coefficient `0.000280`, |coef| `0.000280`
- `lag_05__T_place_TUNNELS`: coefficient `-0.000279`, |coef| `0.000279`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000279`, |coef| `0.000279`
- `lag_11__T3__duck_amount`: coefficient `0.000277`, |coef| `0.000277`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000259`, |coef| `0.000259`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000258`, |coef| `0.000258`
- `lag_15__CT2__flash_duration`: coefficient `0.000256`, |coef| `0.000256`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__centroid_distance_xy`: coefficient `-0.000248`, |coef| `0.000248`
- `lag_00__T_kills_last_3s`: coefficient `-0.000243`, |coef| `0.000243`

## Top 10 utility ridge features

- `lag_14__CT1__flash_duration`: coefficient `0.000422` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `0.000342` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000256` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000219` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000207` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000172` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000169` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000160` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000153` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.000150` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T_place_IVY`: coefficient `0.000497` (raises CT win probability)
- `lag_05__T_place_IVY`: coefficient `0.000416` (raises CT win probability)
- `lag_06__T_place_TUNNELS`: coefficient `-0.000385` (lowers CT win probability)
- `lag_05__T4__is_scoped`: coefficient `0.000280` (raises CT win probability)
- `lag_05__T_place_TUNNELS`: coefficient `-0.000279` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000279` (lowers CT win probability)
- `lag_11__T3__duck_amount`: coefficient `0.000277` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000259` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000258` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000251` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `103212`, seconds `0.50`, LSTM delta `-0.0348`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001333`
- `lag_01__T_place_TSPAWN`: contribution `-0.001144`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001041`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001032`
- `lag_01__centroid_distance_xy`: contribution `-0.000967`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000683`
- `lag_01__molly_inv_diff`: contribution `-0.000471`
- `lag_01__flash_inv_diff`: contribution `-0.000459`
- `lag_01__smoke_inv_diff`: contribution `-0.000408`
- `lag_01__T_utility_inv`: contribution `-0.000335`

### tick `105196`, seconds `31.50`, LSTM delta `-0.0335`

Top all feature movements:
- `lag_06__T_place_IVY`: contribution `-0.005309`
- `lag_14__CT1__flash_duration`: contribution `-0.002869`
- `lag_06__T_place_TUNNELS`: contribution `-0.002158`
- `lag_05__T4__is_scoped`: contribution `-0.001300`
- `lag_11__T3__duck_amount`: contribution `-0.000811`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.002869`

### tick `105164`, seconds `31.00`, LSTM delta `-0.0241`

Top all feature movements:
- `lag_05__T_place_IVY`: contribution `-0.004448`
- `lag_13__CT1__flash_duration`: contribution `-0.002323`
- `lag_05__T_place_TUNNELS`: contribution `-0.001563`
- `lag_15__CT2__flash_duration`: contribution `-0.001524`
- `lag_04__T4__is_scoped`: contribution `-0.000952`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.002323`
- `lag_15__CT2__flash_duration`: contribution `-0.001524`
- `lag_15__CT_flash_duration_sum`: contribution `-0.000415`
- `lag_13__CT_flash_duration_sum`: contribution `-0.000336`

### tick `105100`, seconds `30.00`, LSTM delta `+0.0133`

Top all feature movements:
- `lag_03__T_place_IVY`: contribution `+0.001631`
- `lag_11__T3__duck_amount`: contribution `+0.000796`
- `lag_15__CT4__flash_duration`: contribution `+0.000755`
- `lag_11__CT1__flash_duration`: contribution `+0.000694`
- `lag_13__CT2__flash_duration`: contribution `+0.000640`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `+0.000755`
- `lag_11__CT1__flash_duration`: contribution `+0.000694`
- `lag_13__CT2__flash_duration`: contribution `+0.000640`
- `lag_15__CT_flash_duration_sum`: contribution `-0.000345`
- `lag_13__CT_flash_duration_sum`: contribution `-0.000297`

### tick `105004`, seconds `28.50`, LSTM delta `-0.0128`

Top all feature movements:
- `lag_00__T_place_IVY`: contribution `-0.002497`
- `lag_08__CT1__flash_duration`: contribution `-0.000872`
- `lag_00__T_place_TUNNELS`: contribution `-0.000523`
- `lag_12__CT4__flash_duration`: contribution `-0.000501`
- `lag_13__CT2__is_walking`: contribution `-0.000479`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `-0.000872`
- `lag_12__CT4__flash_duration`: contribution `-0.000501`
- `lag_10__CT2__flash_duration`: contribution `-0.000372`
