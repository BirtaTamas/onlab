# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `146676`, seconds `84.00`, LSTM `0.9289`, delta `+0.0256`
- tick `143156`, seconds `29.00`, LSTM `0.9085`, delta `-0.0245`
- tick `144660`, seconds `52.50`, LSTM `0.9100`, delta `+0.0237`
- tick `141972`, seconds `10.50`, LSTM `0.9520`, delta `+0.0204`
- tick `141332`, seconds `0.50`, LSTM `0.9117`, delta `+0.0203`
- tick `142068`, seconds `12.00`, LSTM `0.9411`, delta `-0.0198`
- tick `144724`, seconds `53.50`, LSTM `0.8926`, delta `-0.0197`
- tick `144404`, seconds `48.50`, LSTM `0.8897`, delta `-0.0173`
- tick `141812`, seconds `8.00`, LSTM `0.9193`, delta `+0.0171`
- tick `146708`, seconds `84.50`, LSTM `0.9457`, delta `+0.0167`

## Top 15 local ridge features

- `lag_00__T_place_BALCONY`: coefficient `0.000881`, |coef| `0.000881`
- `lag_00__T_place_KITCHEN`: coefficient `0.000437`, |coef| `0.000437`
- `lag_00__CT1__is_walking`: coefficient `-0.000412`, |coef| `0.000412`
- `lag_02__T_place_ARCH`: coefficient `0.000356`, |coef| `0.000356`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000303`, |coef| `0.000303`
- `lag_07__T_place_ARCH`: coefficient `0.000292`, |coef| `0.000292`
- `lag_00__T_place_ARCH`: coefficient `-0.000287`, |coef| `0.000287`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000277`, |coef| `0.000277`
- `lag_06__T_place_ARCH`: coefficient `0.000277`, |coef| `0.000277`
- `lag_00__T5__is_walking`: coefficient `-0.000240`, |coef| `0.000240`
- `lag_03__T_place_ARCH`: coefficient `0.000240`, |coef| `0.000240`
- `lag_00__CT_walking_count`: coefficient `-0.000231`, |coef| `0.000231`
- `lag_01__CT_place_BALCONY`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_00__CT_place_RUINS`: coefficient `-0.000216`, |coef| `0.000216`
- `lag_09__T3__is_walking`: coefficient `-0.000210`, |coef| `0.000210`

## Top 10 utility ridge features

- `lag_08__CT_A_site_active_infernos`: coefficient `0.000195` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.000190` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000189` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000186` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000171` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.000151` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000150` (lowers CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.000148` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000145` (lowers CT win probability)
- `lag_08__CT_active_infernos`: coefficient `0.000140` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BALCONY`: coefficient `0.000881` (raises CT win probability)
- `lag_00__T_place_KITCHEN`: coefficient `0.000437` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `-0.000412` (lowers CT win probability)
- `lag_02__T_place_ARCH`: coefficient `0.000356` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000303` (raises CT win probability)
- `lag_07__T_place_ARCH`: coefficient `0.000292` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.000287` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000277` (lowers CT win probability)
- `lag_06__T_place_ARCH`: coefficient `0.000277` (raises CT win probability)
- `lag_00__T5__is_walking`: coefficient `-0.000240` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `146676`, seconds `84.00`, LSTM delta `+0.0256`

Top all feature movements:
- `lag_02__T_place_ARCH`: contribution `+0.003316`
- `lag_00__T_place_ARCH`: contribution `+0.002669`
- `lag_06__T_place_ARCH`: contribution `+0.002577`
- `lag_00__CT_place_BALCONY`: contribution `+0.001779`
- `lag_08__T4__flash_duration`: contribution `+0.001496`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.001496`
- `lag_08__T1__flash_duration`: contribution `+0.001475`
- `lag_08__T_flash_duration_sum`: contribution `+0.001217`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.000657`
- `lag_07__CT_active_infernos`: contribution `+0.000286`

### tick `143156`, seconds `29.00`, LSTM delta `-0.0245`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.012120`
- `lag_00__T_place_DECK`: contribution `-0.004984`
- `lag_02__T_place_BALCONY`: contribution `-0.000921`
- `lag_02__T_place_SECONDMID`: contribution `-0.000581`
- `lag_00__T5__is_walking`: contribution `+0.000557`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144660`, seconds `52.50`, LSTM delta `+0.0237`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.012120`
- `lag_08__T_place_BALCONY`: contribution `+0.001042`
- `lag_06__T2__duck_amount`: contribution `+0.000798`
- `lag_12__T2__duck_amount`: contribution `+0.000649`
- `lag_07__T2__duck_amount`: contribution `-0.000634`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141972`, seconds `10.50`, LSTM delta `+0.0204`

Top all feature movements:
- `lag_00__T_place_KITCHEN`: contribution `+0.013949`
- `lag_00__T_place_UPSTAIRS`: contribution `-0.005111`
- `lag_05__T_place_UPSTAIRS`: contribution `+0.003447`
- `lag_04__T_place_UPSTAIRS`: contribution `+0.000959`
- `lag_08__CT_place_RUINS`: contribution `+0.000668`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141332`, seconds `0.50`, LSTM delta `+0.0203`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000714`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000506`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000483`
- `lag_01__T_place_TSPAWN`: contribution `+0.000457`
- `lag_01__utility_inv_diff`: contribution `+0.000399`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000399`
- `lag_01__smoke_inv_diff`: contribution `+0.000378`
- `lag_01__molly_inv_diff`: contribution `+0.000281`
- `lag_01__CT_molly_inv`: contribution `+0.000189`
- `lag_01__CT_utility_inv`: contribution `+0.000188`
