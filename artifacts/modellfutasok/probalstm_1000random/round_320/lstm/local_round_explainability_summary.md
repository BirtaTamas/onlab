# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `6245`, seconds `12.00`, LSTM `0.6740`, delta `+0.2316`
- tick `6373`, seconds `14.00`, LSTM `0.8658`, delta `+0.1999`
- tick `5957`, seconds `7.50`, LSTM `0.4186`, delta `-0.0803`
- tick `8901`, seconds `53.50`, LSTM `0.8956`, delta `-0.0730`
- tick `6309`, seconds `13.00`, LSTM `0.7283`, delta `+0.0635`
- tick `6341`, seconds `13.50`, LSTM `0.6659`, delta `-0.0624`
- tick `6693`, seconds `19.00`, LSTM `0.9557`, delta `+0.0516`
- tick `9221`, seconds `58.50`, LSTM `0.8272`, delta `+0.0465`
- tick `9733`, seconds `66.50`, LSTM `0.8393`, delta `+0.0452`
- tick `10629`, seconds `80.50`, LSTM `0.8121`, delta `-0.0446`

## Top 15 local ridge features

- `lag_00__T_duck_amount_mean`: coefficient `-0.001938`, |coef| `0.001938`
- `lag_10__CT_place_BDOORS`: coefficient `0.001435`, |coef| `0.001435`
- `lag_02__CT_place_HOLE`: coefficient `0.001389`, |coef| `0.001389`
- `lag_00__damage_diff_last_5s`: coefficient `0.001237`, |coef| `0.001237`
- `lag_00__T4__flash_duration`: coefficient `-0.001216`, |coef| `0.001216`
- `lag_00__T5__duck_amount`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_06__CT_place_HOLE`: coefficient `0.001193`, |coef| `0.001193`
- `lag_01__CT_place_HOLE`: coefficient `-0.001180`, |coef| `0.001180`
- `lag_00__kill_diff_last_3s`: coefficient `0.001152`, |coef| `0.001152`
- `lag_05__CT_place_HOLE`: coefficient `-0.001115`, |coef| `0.001115`
- `lag_00__CT2__is_walking`: coefficient `-0.001096`, |coef| `0.001096`
- `lag_10__CT_place_OUTSIDELONG`: coefficient `0.001094`, |coef| `0.001094`
- `lag_00__T_flashed_players`: coefficient `-0.001046`, |coef| `0.001046`
- `lag_00__CT_kills_last_3s`: coefficient `0.001014`, |coef| `0.001014`
- `lag_00__CT_damage_last_5s`: coefficient `0.000980`, |coef| `0.000980`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `-0.001216` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.000968` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.000813` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.000774` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000764` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000745` (raises CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.000696` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `0.000693` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000665` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.000645` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_duck_amount_mean`: coefficient `-0.001938` (lowers CT win probability)
- `lag_10__CT_place_BDOORS`: coefficient `0.001435` (raises CT win probability)
- `lag_02__CT_place_HOLE`: coefficient `0.001389` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001237` (raises CT win probability)
- `lag_00__T5__duck_amount`: coefficient `-0.001210` (lowers CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `0.001193` (raises CT win probability)
- `lag_01__CT_place_HOLE`: coefficient `-0.001180` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001152` (raises CT win probability)
- `lag_05__CT_place_HOLE`: coefficient `-0.001115` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001096` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `6245`, seconds `12.00`, LSTM delta `+0.2316`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `+0.015508`
- `lag_10__CT_place_BDOORS`: contribution `+0.013802`
- `lag_01__CT_place_HOLE`: contribution `+0.013173`
- `lag_00__T4__flash_duration`: contribution `+0.007961`
- `lag_07__T4__flash_duration`: contribution `+0.005705`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.007961`
- `lag_07__T4__flash_duration`: contribution `+0.005705`
- `lag_05__CT_active_infernos`: contribution `+0.003435`
- `lag_09__CT4__flash_duration`: contribution `+0.003433`
- `lag_07__T_flash_duration_sum`: contribution `+0.003423`

### tick `6373`, seconds `14.00`, LSTM delta `+0.1999`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `+0.013315`
- `lag_05__CT_place_HOLE`: contribution `+0.012452`
- `lag_14__CT_place_BDOORS`: contribution `+0.009312`
- `lag_04__T4__flash_duration`: contribution `+0.005062`
- `lag_13__CT4__flash_duration`: contribution `+0.004329`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.005062`
- `lag_13__CT4__flash_duration`: contribution `+0.004329`
- `lag_11__T4__flash_duration`: contribution `+0.004083`
- `lag_02__T5__flash_duration`: contribution `+0.002998`
- `lag_11__T_flash_duration_sum`: contribution `+0.002935`

### tick `5957`, seconds `7.50`, LSTM delta `-0.0803`

Top all feature movements:
- `lag_10__T_place_OUTSIDELONG`: contribution `-0.004581`
- `lag_01__CT_place_BDOORS`: contribution `-0.003722`
- `lag_12__CT_place_UNDERA`: contribution `-0.002633`
- `lag_06__CT_place_UNDERA`: contribution `-0.002589`
- `lag_15__T_place_TSPAWN`: contribution `-0.002446`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.001617`
- `lag_15__CT1__flash`: contribution `-0.001297`
- `lag_15__CT1__utility_total`: contribution `-0.000965`
- `lag_15__T2__smoke`: contribution `-0.000929`

### tick `8901`, seconds `53.50`, LSTM delta `-0.0730`

Top all feature movements:
- `lag_10__CT_place_OUTSIDELONG`: contribution `-0.011095`
- `lag_00__T_duck_amount_mean`: contribution `-0.005586`
- `lag_10__CT_place_LONGDOORS`: contribution `-0.003821`
- `lag_03__CT_place_ARAMP`: contribution `-0.002989`
- `lag_00__damage_diff_last_5s`: contribution `-0.002792`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6309`, seconds `13.00`, LSTM delta `+0.0635`

Top all feature movements:
- `lag_12__CT_place_BDOORS`: contribution `+0.005509`
- `lag_04__CT_place_HOLE`: contribution `+0.003432`
- `lag_00__CT2__duck_amount`: contribution `+0.002310`
- `lag_12__T_place_LONGDOORS`: contribution `+0.002170`
- `lag_14__T_place_LONGDOORS`: contribution `+0.002121`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `-0.001772`
- `lag_00__T5__flash_duration`: contribution `+0.001600`
- `lag_00__T_flash_duration_sum`: contribution `+0.001400`
