# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `18724`, seconds `30.00`, LSTM `0.7653`, delta `+0.2824`
- tick `18500`, seconds `26.50`, LSTM `0.4583`, delta `-0.2108`
- tick `20804`, seconds `62.50`, LSTM `0.9335`, delta `+0.1610`
- tick `21348`, seconds `71.00`, LSTM `0.7932`, delta `-0.1167`
- tick `18692`, seconds `29.50`, LSTM `0.4829`, delta `+0.1144`
- tick `18756`, seconds `30.50`, LSTM `0.8609`, delta `+0.0956`
- tick `19556`, seconds `43.00`, LSTM `0.6507`, delta `-0.0683`
- tick `19812`, seconds `47.00`, LSTM `0.7164`, delta `+0.0671`
- tick `18660`, seconds `29.00`, LSTM `0.3685`, delta `-0.0564`
- tick `21540`, seconds `74.00`, LSTM `0.8218`, delta `+0.0514`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003544`, |coef| `0.003544`
- `lag_00__damage_diff_last_5s`: coefficient `0.002708`, |coef| `0.002708`
- `lag_00__T_kills_last_3s`: coefficient `-0.002306`, |coef| `0.002306`
- `lag_00__CT_kills_last_3s`: coefficient `0.002149`, |coef| `0.002149`
- `lag_00__CT_place_BALCONY`: coefficient `-0.002104`, |coef| `0.002104`
- `lag_05__T3__flash_duration`: coefficient `-0.002098`, |coef| `0.002098`
- `lag_06__CT1__duck_amount`: coefficient `-0.002044`, |coef| `0.002044`
- `lag_09__T4__is_walking`: coefficient `-0.001954`, |coef| `0.001954`
- `lag_12__CT4__is_walking`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_12__T3__flash_duration`: coefficient `0.001802`, |coef| `0.001802`
- `lag_01__CT_place_TOPOFMID`: coefficient `0.001778`, |coef| `0.001778`
- `lag_00__CT_damage_last_5s`: coefficient `0.001775`, |coef| `0.001775`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001755`, |coef| `0.001755`
- `lag_07__CT_place_TOPOFMID`: coefficient `0.001701`, |coef| `0.001701`
- `lag_05__CT5__duck_amount`: coefficient `-0.001679`, |coef| `0.001679`

## Top 10 utility ridge features

- `lag_05__T3__flash_duration`: coefficient `-0.002098` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.001802` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.001440` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.001281` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.001276` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.001269` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001252` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.001186` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001165` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.001016` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003544` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002708` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002306` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002149` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.002104` (lowers CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `-0.002044` (lowers CT win probability)
- `lag_09__T4__is_walking`: coefficient `-0.001954` (lowers CT win probability)
- `lag_12__CT4__is_walking`: coefficient `-0.001812` (lowers CT win probability)
- `lag_01__CT_place_TOPOFMID`: coefficient `0.001778` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001775` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18724`, seconds `30.00`, LSTM delta `+0.2824`

Top all feature movements:
- `lag_12__T3__flash_duration`: contribution `+0.011098`
- `lag_00__damage_diff_last_5s`: contribution `+0.008799`
- `lag_00__kill_diff_last_3s`: contribution `+0.008529`
- `lag_00__T3__flash_duration`: contribution `+0.008030`
- `lag_06__CT1__duck_amount`: contribution `+0.007799`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `+0.011098`
- `lag_00__T3__flash_duration`: contribution `+0.008030`
- `lag_02__T2__flash_duration`: contribution `+0.007192`
- `lag_02__T1__flash_duration`: contribution `+0.007145`
- `lag_02__T_flash_duration_sum`: contribution `+0.005238`

### tick `18500`, seconds `26.50`, LSTM delta `-0.2108`

Top all feature movements:
- `lag_05__T3__flash_duration`: contribution `-0.012919`
- `lag_00__kill_diff_last_3s`: contribution `-0.008529`
- `lag_00__T_kills_last_3s`: contribution `-0.007305`
- `lag_05__CT5__duck_amount`: contribution `-0.006340`
- `lag_07__CT_place_TOPOFMID`: contribution `-0.006172`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.012919`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.004521`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.004074`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.003455`

### tick `20804`, seconds `62.50`, LSTM delta `+0.1610`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008529`
- `lag_06__CT1__duck_amount`: contribution `+0.007726`
- `lag_01__CT_place_TOPOFMID`: contribution `+0.006451`
- `lag_00__CT_kills_last_3s`: contribution `+0.006205`
- `lag_09__T2__duck_amount`: contribution `+0.005632`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21348`, seconds `71.00`, LSTM delta `-0.1167`

Top all feature movements:
- `lag_00__T_duck_amount_mean`: contribution `-0.008840`
- `lag_00__kill_diff_last_3s`: contribution `-0.008529`
- `lag_00__T_kills_last_3s`: contribution `-0.007305`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.005870`
- `lag_10__CT_place_TOPOFMID`: contribution `-0.004477`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.002399`

### tick `18692`, seconds `29.50`, LSTM delta `+0.1144`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008529`
- `lag_06__CT1__duck_amount`: contribution `-0.007799`
- `lag_00__T_kills_last_3s`: contribution `+0.007305`
- `lag_11__T3__flash_duration`: contribution `+0.006254`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006095`

Top utility-only movements:
- `lag_11__T3__flash_duration`: contribution `+0.006254`
- `lag_01__T2__flash_duration`: contribution `+0.003084`
- `lag_12__CT_A_site_active_infernos`: contribution `+0.003035`
- `lag_01__T1__flash_duration`: contribution `+0.002957`
