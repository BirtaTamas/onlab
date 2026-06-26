# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `16702`, seconds `61.00`, LSTM `0.9694`, delta `+0.0345`
- tick `12830`, seconds `0.50`, LSTM `0.9095`, delta `+0.0228`
- tick `15806`, seconds `47.00`, LSTM `0.9621`, delta `+0.0201`
- tick `14014`, seconds `19.00`, LSTM `0.9577`, delta `+0.0146`
- tick `15230`, seconds `38.00`, LSTM `0.9603`, delta `-0.0121`
- tick `17406`, seconds `72.00`, LSTM `0.9752`, delta `+0.0120`
- tick `14174`, seconds `21.50`, LSTM `0.9736`, delta `+0.0118`
- tick `12958`, seconds `2.50`, LSTM `0.8900`, delta `-0.0101`
- tick `15582`, seconds `43.50`, LSTM `0.9538`, delta `+0.0101`
- tick `13662`, seconds `13.50`, LSTM `0.9366`, delta `+0.0101`

## Top 15 local ridge features

- `lag_12__T_place_ARCH`: coefficient `0.000720`, |coef| `0.000720`
- `lag_00__T_place_ARCH`: coefficient `-0.000572`, |coef| `0.000572`
- `lag_05__CT_place_PIT`: coefficient `0.000382`, |coef| `0.000382`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000341`, |coef| `0.000341`
- `lag_00__CT1__is_walking`: coefficient `-0.000316`, |coef| `0.000316`
- `lag_00__CT_kills_last_3s`: coefficient `0.000297`, |coef| `0.000297`
- `lag_14__T_place_KITCHEN`: coefficient `0.000269`, |coef| `0.000269`
- `lag_05__CT_place_TOPOFMID`: coefficient `0.000262`, |coef| `0.000262`
- `lag_00__kill_diff_last_3s`: coefficient `0.000260`, |coef| `0.000260`
- `lag_14__T4__duck_amount`: coefficient `-0.000257`, |coef| `0.000257`
- `lag_00__CT_place_LIBRARY`: coefficient `-0.000248`, |coef| `0.000248`
- `lag_15__T4__duck_amount`: coefficient `0.000243`, |coef| `0.000243`
- `lag_13__T_place_ARCH`: coefficient `0.000238`, |coef| `0.000238`
- `lag_04__T3__is_walking`: coefficient `0.000235`, |coef| `0.000235`
- `lag_13__T_place_DECK`: coefficient `-0.000226`, |coef| `0.000226`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000205` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000186` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000185` (raises CT win probability)
- `lag_01__CT_molly_inv`: coefficient `0.000158` (raises CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `-0.000155` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.000154` (raises CT win probability)
- `lag_01__CT_utility_inv`: coefficient `0.000141` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000140` (raises CT win probability)
- `lag_01__CT_active_smokes`: coefficient `-0.000133` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000131` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_ARCH`: coefficient `0.000720` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.000572` (lowers CT win probability)
- `lag_05__CT_place_PIT`: coefficient `0.000382` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000341` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `-0.000316` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000297` (raises CT win probability)
- `lag_14__T_place_KITCHEN`: coefficient `0.000269` (raises CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `0.000262` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000260` (raises CT win probability)
- `lag_14__T4__duck_amount`: coefficient `-0.000257` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16702`, seconds `61.00`, LSTM delta `+0.0345`

Top all feature movements:
- `lag_12__T_place_ARCH`: contribution `+0.006697`
- `lag_00__T_place_ARCH`: contribution `+0.005326`
- `lag_05__CT_place_PIT`: contribution `+0.001643`
- `lag_14__T4__duck_amount`: contribution `+0.000951`
- `lag_15__T4__duck_amount`: contribution `+0.000897`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `12830`, seconds `0.50`, LSTM delta `+0.0228`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.001062`
- `lag_01__T_place_TSPAWN`: contribution `+0.000852`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000846`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000768`
- `lag_01__utility_inv_diff`: contribution `+0.000726`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000726`
- `lag_01__molly_inv_diff`: contribution `+0.000610`
- `lag_01__smoke_inv_diff`: contribution `+0.000596`
- `lag_01__CT_molly_inv`: contribution `+0.000473`
- `lag_01__CT_utility_inv`: contribution `+0.000366`

### tick `15806`, seconds `47.00`, LSTM delta `+0.0201`

Top all feature movements:
- `lag_13__T_place_DECK`: contribution `+0.005493`
- `lag_00__CT_place_LOWERMID`: contribution `+0.004640`
- `lag_09__CT_place_TRAMP`: contribution `+0.002036`
- `lag_00__CT_place_TRAMP`: contribution `+0.001306`
- `lag_00__CT_macro_MID`: contribution `+0.000496`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14014`, seconds `19.00`, LSTM delta `+0.0146`

Top all feature movements:
- `lag_00__T_place_UPSTAIRS`: contribution `+0.005748`
- `lag_07__T5__flash_duration`: contribution `+0.001073`
- `lag_05__CT_place_TOPOFMID`: contribution `+0.000950`
- `lag_05__CT_place_ARCH`: contribution `+0.000703`
- `lag_07__CT1__flash_duration`: contribution `+0.000435`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `+0.001073`
- `lag_07__CT1__flash_duration`: contribution `+0.000435`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.000308`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.000273`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.000230`

### tick `15230`, seconds `38.00`, LSTM delta `-0.0121`

Top all feature movements:
- `lag_00__T_place_KITCHEN`: contribution `-0.005690`
- `lag_03__T_place_KITCHEN`: contribution `-0.004072`
- `lag_00__T_place_DECK`: contribution `-0.000812`
- `lag_09__CT1__is_walking`: contribution `-0.000488`
- `lag_04__T2__is_walking`: contribution `-0.000417`

Top utility-only movements:
- No utility movement among the top local contributors.
