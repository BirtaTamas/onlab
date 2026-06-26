# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `22375`, seconds `14.00`, LSTM `0.7665`, delta `+0.1834`
- tick `22535`, seconds `16.50`, LSTM `0.9237`, delta `+0.1219`
- tick `22279`, seconds `12.50`, LSTM `0.5872`, delta `+0.1092`
- tick `24423`, seconds `46.00`, LSTM `0.9590`, delta `+0.0390`
- tick `24199`, seconds `42.50`, LSTM `0.9360`, delta `+0.0299`
- tick `22247`, seconds `12.00`, LSTM `0.4781`, delta `+0.0262`
- tick `22311`, seconds `13.00`, LSTM `0.5642`, delta `-0.0230`
- tick `22407`, seconds `14.50`, LSTM `0.7892`, delta `+0.0227`
- tick `22503`, seconds `16.00`, LSTM `0.8018`, delta `+0.0210`
- tick `22343`, seconds `13.50`, LSTM `0.5831`, delta `+0.0189`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `-0.002137`, |coef| `0.002137`
- `lag_02__T_flashes_last_5s`: coefficient `0.001682`, |coef| `0.001682`
- `lag_14__CT_place_TOPOFMID`: coefficient `0.001159`, |coef| `0.001159`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001097`, |coef| `0.001097`
- `lag_00__CT_kills_last_3s`: coefficient `0.001073`, |coef| `0.001073`
- `lag_14__CT5__flash_duration`: coefficient `0.001014`, |coef| `0.001014`
- `lag_07__T_flashes_last_5s`: coefficient `0.000921`, |coef| `0.000921`
- `lag_15__CT_place_HOUSE`: coefficient `-0.000920`, |coef| `0.000920`
- `lag_00__kill_diff_last_3s`: coefficient `0.000895`, |coef| `0.000895`
- `lag_00__CT_damage_last_5s`: coefficient `0.000839`, |coef| `0.000839`
- `lag_00__damage_diff_last_5s`: coefficient `0.000834`, |coef| `0.000834`
- `lag_02__CT5__shots_fired`: coefficient `-0.000793`, |coef| `0.000793`
- `lag_11__CT_place_MIDDLE`: coefficient `0.000790`, |coef| `0.000790`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.000790`, |coef| `0.000790`
- `lag_13__T_place_WATER`: coefficient `-0.000776`, |coef| `0.000776`

## Top 10 utility ridge features

- `lag_02__T_flashes_last_5s`: coefficient `0.001682` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `0.001014` (raises CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `0.000921` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000713` (raises CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `0.000607` (raises CT win probability)
- `lag_09__CT4__flash`: coefficient `-0.000573` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000559` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000522` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `0.000482` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000475` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `-0.002137` (lowers CT win probability)
- `lag_14__CT_place_TOPOFMID`: coefficient `0.001159` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001097` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001073` (raises CT win probability)
- `lag_15__CT_place_HOUSE`: coefficient `-0.000920` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000895` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000839` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000834` (raises CT win probability)
- `lag_02__CT5__shots_fired`: coefficient `-0.000793` (lowers CT win probability)
- `lag_11__CT_place_MIDDLE`: coefficient `0.000790` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22375`, seconds `14.00`, LSTM delta `+0.1834`

Top all feature movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.015236`
- `lag_14__CT_place_TOPOFMID`: contribution `+0.008411`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008228`
- `lag_02__CT5__shots_fired`: contribution `+0.006293`
- `lag_14__CT5__flash_duration`: contribution `+0.005139`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.015236`
- `lag_14__CT5__flash_duration`: contribution `+0.005139`
- `lag_14__CT_flash_duration_sum`: contribution `+0.002011`

### tick `22535`, seconds `16.50`, LSTM delta `+0.1219`

Top all feature movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.008344`
- `lag_07__CT_shots_fired_sum`: contribution `+0.006345`
- `lag_07__CT5__shots_fired`: contribution `+0.005474`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003810`
- `lag_00__CT_kills_last_3s`: contribution `+0.003098`

Top utility-only movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.008344`
- `lag_14__CT5__flash_duration`: contribution `+0.002211`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.001276`

### tick `22279`, seconds `12.50`, LSTM delta `+0.1092`

Top all feature movements:
- `lag_11__CT_place_HOUSE`: contribution `+0.005009`
- `lag_13__T_place_WATER`: contribution `+0.004427`
- `lag_15__T_place_TUNNEL`: contribution `+0.004263`
- `lag_11__CT5__flash_duration`: contribution `+0.003616`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003451`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.003616`

### tick `24423`, seconds `46.00`, LSTM delta `+0.0390`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.003810`
- `lag_00__CT_kills_last_3s`: contribution `+0.003098`
- `lag_03__CT3__shots_fired`: contribution `+0.002962`
- `lag_04__CT3__duck_amount`: contribution `+0.002385`
- `lag_00__kill_diff_last_3s`: contribution `+0.002153`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24199`, seconds `42.50`, LSTM delta `+0.0299`

Top all feature movements:
- `lag_03__CT_place_MAINHALL`: contribution `+0.004204`
- `lag_07__CT_place_MAINHALL`: contribution `+0.003746`
- `lag_14__CT_place_HOUSE`: contribution `-0.002488`
- `lag_12__CT_place_HOUSE`: contribution `+0.002483`
- `lag_00__T_bomb_zone_count`: contribution `+0.002099`

Top utility-only movements:
- `lag_10__T_active_infernos`: contribution `-0.000827`
