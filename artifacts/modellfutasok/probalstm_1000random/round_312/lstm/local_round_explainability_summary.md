# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `2977`, seconds `46.50`, LSTM `0.5153`, delta `+0.3324`
- tick `2049`, seconds `32.00`, LSTM `0.5041`, delta `+0.2411`
- tick `3041`, seconds `47.50`, LSTM `0.7130`, delta `+0.1993`
- tick `2785`, seconds `43.50`, LSTM `0.2328`, delta `-0.1518`
- tick `3201`, seconds `50.00`, LSTM `0.9215`, delta `+0.1444`
- tick `1345`, seconds `21.00`, LSTM `0.2247`, delta `-0.0922`
- tick `1313`, seconds `20.50`, LSTM `0.3169`, delta `-0.0906`
- tick `1857`, seconds `29.00`, LSTM `0.3193`, delta `+0.0605`
- tick `1281`, seconds `20.00`, LSTM `0.4075`, delta `-0.0524`
- tick `1697`, seconds `26.50`, LSTM `0.2800`, delta `+0.0524`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003510`, |coef| `0.003510`
- `lag_00__T_place_ARAMP`: coefficient `-0.003050`, |coef| `0.003050`
- `lag_00__CT_kills_last_3s`: coefficient `0.002866`, |coef| `0.002866`
- `lag_00__T_place_SIDE`: coefficient `-0.002543`, |coef| `0.002543`
- `lag_00__T_place_LONGA`: coefficient `-0.002346`, |coef| `0.002346`
- `lag_00__damage_diff_last_5s`: coefficient `0.002215`, |coef| `0.002215`
- `lag_15__T_place_ARAMP`: coefficient `-0.002139`, |coef| `0.002139`
- `lag_01__T_place_ARAMP`: coefficient `-0.002114`, |coef| `0.002114`
- `lag_03__T_place_ARAMP`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `-0.001823`, |coef| `0.001823`
- `lag_15__T_bomb_zone_count`: coefficient `0.001804`, |coef| `0.001804`
- `lag_01__T_place_SIDE`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_00__CT_damage_last_5s`: coefficient `0.001718`, |coef| `0.001718`
- `lag_00__CT2__duck_amount`: coefficient `-0.001604`, |coef| `0.001604`
- `lag_04__T_bomb_zone_count`: coefficient `-0.001565`, |coef| `0.001565`

## Top 10 utility ridge features

- `lag_03__T3__flash_duration`: coefficient `-0.001537` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.001351` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.001048` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001018` (lowers CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `-0.000880` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.000767` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `-0.000766` (lowers CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000703` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000669` (raises CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000663` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003510` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.003050` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002866` (raises CT win probability)
- `lag_00__T_place_SIDE`: coefficient `-0.002543` (lowers CT win probability)
- `lag_00__T_place_LONGA`: coefficient `-0.002346` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002215` (raises CT win probability)
- `lag_15__T_place_ARAMP`: coefficient `-0.002139` (lowers CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `-0.002114` (lowers CT win probability)
- `lag_03__T_place_ARAMP`: coefficient `-0.001878` (lowers CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `-0.001823` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2977`, seconds `46.50`, LSTM delta `+0.3324`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.025345`
- `lag_15__T_place_ARAMP`: contribution `+0.019353`
- `lag_01__T_place_ARAMP`: contribution `+0.019123`
- `lag_00__CT_kills_last_3s`: contribution `+0.016552`
- `lag_15__T_bomb_zone_count`: contribution `+0.010505`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.006172`
- `lag_09__T3__flash_duration`: contribution `+0.004785`

### tick `2049`, seconds `32.00`, LSTM delta `+0.2411`

Top all feature movements:
- `lag_11__T_place_SIDE`: contribution `+0.029978`
- `lag_00__T_place_ARAMP`: contribution `+0.027597`
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.021013`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.010163`
- `lag_00__kill_diff_last_3s`: contribution `+0.008448`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `+0.003245`

### tick `3041`, seconds `47.50`, LSTM delta `+0.1993`

Top all feature movements:
- `lag_03__T_place_ARAMP`: contribution `+0.016994`
- `lag_00__T_place_LONGA`: contribution `+0.009994`
- `lag_02__kill_diff_last_3s`: contribution `+0.009775`
- `lag_00__kill_diff_last_3s`: contribution `+0.008448`
- `lag_00__CT_kills_last_3s`: contribution `+0.008276`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `+0.007019`
- `lag_11__T3__flash_duration`: contribution `+0.003505`

### tick `2785`, seconds `43.50`, LSTM delta `-0.1518`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008448`
- `lag_03__T3__flash_duration`: contribution `-0.007019`
- `lag_09__T_bomb_zone_count`: contribution `-0.006613`
- `lag_03__T_flashed_players`: contribution `-0.005957`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.005150`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.007019`
- `lag_03__T_flash_duration_sum`: contribution `-0.002376`

### tick `3201`, seconds `50.00`, LSTM delta `+0.1444`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.027597`
- `lag_00__kill_diff_last_3s`: contribution `+0.008448`
- `lag_00__CT_kills_last_3s`: contribution `+0.008276`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.006835`
- `lag_00__T_flash_alpha_mean`: contribution `+0.006175`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.006175`
