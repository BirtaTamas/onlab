# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `28662`, seconds `35.50`, LSTM `0.1568`, delta `+0.0588`
- tick `26422`, seconds `0.50`, LSTM `0.0553`, delta `-0.0569`
- tick `30454`, seconds `63.50`, LSTM `0.1427`, delta `+0.0569`
- tick `32438`, seconds `94.50`, LSTM `0.0223`, delta `-0.0416`
- tick `30326`, seconds `61.50`, LSTM `0.1095`, delta `-0.0415`
- tick `28566`, seconds `34.00`, LSTM `0.1016`, delta `-0.0409`
- tick `28438`, seconds `32.00`, LSTM `0.1392`, delta `-0.0394`
- tick `32086`, seconds `89.00`, LSTM `0.0819`, delta `-0.0383`
- tick `31062`, seconds `73.00`, LSTM `0.1566`, delta `-0.0371`
- tick `29430`, seconds `47.50`, LSTM `0.1823`, delta `-0.0356`

## Top 15 local ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.001705`, |coef| `0.001705`
- `lag_00__CT1__is_walking`: coefficient `0.000984`, |coef| `0.000984`
- `lag_00__T5__duck_amount`: coefficient `-0.000958`, |coef| `0.000958`
- `lag_01__T_place_BALCONY`: coefficient `-0.000816`, |coef| `0.000816`
- `lag_00__T_place_SECONDMID`: coefficient `0.000807`, |coef| `0.000807`
- `lag_09__CT_place_LIBRARY`: coefficient `0.000798`, |coef| `0.000798`
- `lag_09__T2__duck_amount`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_01__T3__is_walking`: coefficient `-0.000757`, |coef| `0.000757`
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000698`, |coef| `0.000698`
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000685`, |coef| `0.000685`
- `lag_00__T2__is_walking`: coefficient `0.000653`, |coef| `0.000653`
- `lag_00__CT4__flash_duration`: coefficient `-0.000644`, |coef| `0.000644`
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000641`, |coef| `0.000641`
- `lag_00__T3__is_walking`: coefficient `-0.000629`, |coef| `0.000629`
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.000621`, |coef| `0.000621`

## Top 10 utility ridge features

- `lag_11__T_B_site_active_infernos`: coefficient `-0.000698` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000685` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000644` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000641` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `-0.000618` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000520` (lowers CT win probability)
- `lag_08__T_active_infernos`: coefficient `-0.000519` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000490` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.000478` (lowers CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000435` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.001705` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000984` (raises CT win probability)
- `lag_00__T5__duck_amount`: coefficient `-0.000958` (lowers CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `-0.000816` (lowers CT win probability)
- `lag_00__T_place_SECONDMID`: coefficient `0.000807` (raises CT win probability)
- `lag_09__CT_place_LIBRARY`: coefficient `0.000798` (raises CT win probability)
- `lag_09__T2__duck_amount`: coefficient `-0.000775` (lowers CT win probability)
- `lag_01__T3__is_walking`: coefficient `-0.000757` (lowers CT win probability)
- `lag_00__T2__is_walking`: coefficient `0.000653` (raises CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000629` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `28662`, seconds `35.50`, LSTM delta `+0.0588`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.023447`
- `lag_03__T_place_BALCONY`: contribution `+0.003759`
- `lag_00__T_place_SECONDMID`: contribution `+0.002642`
- `lag_06__CT_place_LIBRARY`: contribution `+0.002152`
- `lag_05__T5__duck_amount`: contribution `+0.001800`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26422`, seconds `0.50`, LSTM delta `-0.0569`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002637`
- `lag_01__T_place_TSPAWN`: contribution `-0.002109`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.001902`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001872`
- `lag_00__T_velocity_mean`: contribution `-0.001856`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.001902`
- `lag_01__molly_inv_diff`: contribution `-0.000644`
- `lag_01__smoke_inv_diff`: contribution `-0.000604`
- `lag_01__utility_inv_diff`: contribution `-0.000576`
- `lag_01__T_smoke_inv`: contribution `-0.000529`

### tick `30454`, seconds `63.50`, LSTM delta `+0.0569`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.023447`
- `lag_00__T5__duck_amount`: contribution `+0.003639`
- `lag_00__T_place_SECONDMID`: contribution `+0.002642`
- `lag_04__T_place_BALCONY`: contribution `+0.002599`
- `lag_09__T2__duck_amount`: contribution `+0.002116`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.001031`

### tick `32438`, seconds `94.50`, LSTM delta `-0.0416`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.004626`
- `lag_06__T_place_BALCONY`: contribution `-0.003926`
- `lag_03__T_place_BALCONY`: contribution `-0.003759`
- `lag_06__T_place_PIT`: contribution `-0.002718`
- `lag_11__T_place_BALCONY`: contribution `-0.002614`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `-0.001590`
- `lag_08__T_active_infernos`: contribution `-0.001081`
- `lag_08__T_A_site_active_infernos`: contribution `-0.001071`

### tick `30326`, seconds `61.50`, LSTM delta `-0.0415`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.023447`
- `lag_01__T3__is_walking`: contribution `-0.001758`
- `lag_09__CT5__duck_amount`: contribution `-0.001703`
- `lag_00__T_velocity_mean`: contribution `+0.001586`
- `lag_00__T_place_APARTMENTS`: contribution `-0.001375`

Top utility-only movements:
- `lag_14__T_active_infernos`: contribution `-0.000546`
- `lag_09__CT3__flash`: contribution `-0.000450`
