# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `88567`, seconds `33.50`, LSTM `0.7606`, delta `+0.2229`
- tick `88407`, seconds `31.00`, LSTM `0.5204`, delta `+0.1128`
- tick `92471`, seconds `94.50`, LSTM `0.9281`, delta `+0.1077`
- tick `87095`, seconds `10.50`, LSTM `0.2622`, delta `-0.0850`
- tick `87159`, seconds `11.50`, LSTM `0.3205`, delta `+0.0793`
- tick `88599`, seconds `34.00`, LSTM `0.6990`, delta `-0.0617`
- tick `88055`, seconds `25.50`, LSTM `0.3435`, delta `-0.0556`
- tick `88375`, seconds `30.50`, LSTM `0.4076`, delta `+0.0509`
- tick `86487`, seconds `1.00`, LSTM `0.3634`, delta `-0.0455`
- tick `91479`, seconds `79.00`, LSTM `0.8003`, delta `+0.0444`

## Top 15 local ridge features

- `lag_15__T_place_PIPE`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_00__CT_kills_last_3s`: coefficient `0.001436`, |coef| `0.001436`
- `lag_08__T_place_CONSTRUCTION`: coefficient `0.001304`, |coef| `0.001304`
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.001243`, |coef| `0.001243`
- `lag_00__kill_diff_last_3s`: coefficient `0.001115`, |coef| `0.001115`
- `lag_02__CT_place_BRIDGE`: coefficient `0.001074`, |coef| `0.001074`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001012`, |coef| `0.001012`
- `lag_00__damage_diff_last_5s`: coefficient `0.000994`, |coef| `0.000994`
- `lag_00__CT_place_WATER`: coefficient `0.000988`, |coef| `0.000988`
- `lag_12__T_flashes_last_5s`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_06__T_place_LOWERPARK`: coefficient `0.000927`, |coef| `0.000927`
- `lag_03__T_place_LOWERPARK`: coefficient `0.000915`, |coef| `0.000915`
- `lag_10__T_place_ALLEY`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_00__CT_damage_last_5s`: coefficient `0.000907`, |coef| `0.000907`
- `lag_10__T_place_PIPE`: coefficient `-0.000886`, |coef| `0.000886`

## Top 10 utility ridge features

- `lag_12__T_flashes_last_5s`: coefficient `-0.000974` (lowers CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `-0.000852` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000760` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.000741` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000734` (lowers CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `-0.000713` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000700` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000686` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.000636` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000616` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_PIPE`: coefficient `-0.001600` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001436` (raises CT win probability)
- `lag_08__T_place_CONSTRUCTION`: coefficient `0.001304` (raises CT win probability)
- `lag_00__T_place_CONSTRUCTION`: coefficient `-0.001243` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001115` (raises CT win probability)
- `lag_02__CT_place_BRIDGE`: coefficient `0.001074` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001012` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000994` (raises CT win probability)
- `lag_00__CT_place_WATER`: coefficient `0.000988` (raises CT win probability)
- `lag_06__T_place_LOWERPARK`: coefficient `0.000927` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `88567`, seconds `33.50`, LSTM delta `+0.2229`

Top all feature movements:
- `lag_15__T_place_PIPE`: contribution `+0.020434`
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.016203`
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.015444`
- `lag_02__CT_place_BRIDGE`: contribution `+0.012307`
- `lag_00__CT_place_BRIDGE`: contribution `+0.007266`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `+0.004362`
- `lag_13__CT4__flash_duration`: contribution `+0.003550`
- `lag_13__CT2__flash_duration`: contribution `+0.003431`
- `lag_15__CT1__flash_duration`: contribution `+0.002889`

### tick `88407`, seconds `31.00`, LSTM delta `+0.1128`

Top all feature movements:
- `lag_10__T_place_PIPE`: contribution `+0.011313`
- `lag_03__T_place_CONSTRUCTION`: contribution `+0.006956`
- `lag_08__CT_flash_duration_sum`: contribution `+0.004850`
- `lag_08__CT4__flash_duration`: contribution `+0.004846`
- `lag_05__T_place_TSTAIRS`: contribution `+0.004653`

Top utility-only movements:
- `lag_08__CT_flash_duration_sum`: contribution `+0.004850`
- `lag_08__CT4__flash_duration`: contribution `+0.004846`
- `lag_08__CT2__flash_duration`: contribution `+0.004639`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.004222`
- `lag_11__T_utility_damage_last_5s`: contribution `+0.003582`

### tick `92471`, seconds `94.50`, LSTM delta `+0.1077`

Top all feature movements:
- `lag_00__T_place_UPPERPARK`: contribution `+0.004529`
- `lag_00__CT_kills_last_3s`: contribution `+0.004147`
- `lag_15__CT5__is_scoped`: contribution `+0.002856`
- `lag_09__T4__is_scoped`: contribution `+0.002757`
- `lag_00__kill_diff_last_3s`: contribution `+0.002683`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.002083`

### tick `87095`, seconds `10.50`, LSTM delta `-0.0850`

Top all feature movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.008821`
- `lag_08__T_place_TSTAIRS`: contribution `-0.008310`
- `lag_15__CT_place_STAIRS`: contribution `-0.006603`
- `lag_00__CT_place_WATER`: contribution `-0.006003`
- `lag_12__CT_place_STAIRS`: contribution `-0.004626`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.008821`
- `lag_02__T_flashes_last_5s`: contribution `-0.003487`

### tick `87159`, seconds `11.50`, LSTM delta `+0.0793`

Top all feature movements:
- `lag_15__CT_place_STAIRS`: contribution `+0.006603`
- `lag_11__T_flashes_last_5s`: contribution `+0.006463`
- `lag_00__CT_place_WATER`: contribution `+0.006003`
- `lag_10__T_place_TSTAIRS`: contribution `+0.005995`
- `lag_02__CT_place_WATER`: contribution `+0.005090`

Top utility-only movements:
- `lag_11__T_flashes_last_5s`: contribution `+0.006463`
- `lag_04__T_flashes_last_5s`: contribution `+0.004733`
