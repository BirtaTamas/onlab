# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `56164`, seconds `57.50`, LSTM `0.1331`, delta `-0.4340`
- tick `58052`, seconds `87.00`, LSTM `0.7488`, delta `+0.3974`
- tick `59204`, seconds `105.00`, LSTM `0.7546`, delta `+0.3212`
- tick `58980`, seconds `101.50`, LSTM `0.4421`, delta `-0.3091`
- tick `58020`, seconds `86.50`, LSTM `0.3514`, delta `-0.2965`
- tick `57572`, seconds `79.50`, LSTM `0.5380`, delta `+0.2042`
- tick `57700`, seconds `81.50`, LSTM `0.7436`, delta `+0.2027`
- tick `58820`, seconds `99.00`, LSTM `0.7965`, delta `+0.1637`
- tick `59396`, seconds `108.00`, LSTM `0.8613`, delta `+0.1274`
- tick `56580`, seconds `64.00`, LSTM `0.1485`, delta `+0.0966`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.009275`, |coef| `0.009275`
- `lag_00__kill_diff_last_3s`: coefficient `0.005531`, |coef| `0.005531`
- `lag_00__T_kills_last_3s`: coefficient `-0.004601`, |coef| `0.004601`
- `lag_05__CT_defusing_count`: coefficient `-0.003830`, |coef| `0.003830`
- `lag_00__T_damage_last_5s`: coefficient `-0.002881`, |coef| `0.002881`
- `lag_07__CT_defusing_count`: coefficient `-0.002790`, |coef| `0.002790`
- `lag_00__CT_velocity_mean`: coefficient `-0.002648`, |coef| `0.002648`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002565`, |coef| `0.002565`
- `lag_15__T4__flash_duration`: coefficient `-0.002562`, |coef| `0.002562`
- `lag_12__CT_defusing_count`: coefficient `0.002489`, |coef| `0.002489`
- `lag_00__CT_kills_last_3s`: coefficient `0.002442`, |coef| `0.002442`
- `lag_00__damage_diff_last_5s`: coefficient `0.002416`, |coef| `0.002416`
- `lag_15__T2__flash_duration`: coefficient `0.002356`, |coef| `0.002356`
- `lag_12__T4__flash_duration`: coefficient `0.002209`, |coef| `0.002209`
- `lag_00__alive_diff`: coefficient `0.002053`, |coef| `0.002053`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002565` (lowers CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.002562` (lowers CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.002356` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.002209` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `0.001910` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001670` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001656` (lowers CT win probability)
- `lag_13__T_flash_alpha_mean`: coefficient `-0.001408` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001381` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001306` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.009275` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005531` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004601` (lowers CT win probability)
- `lag_05__CT_defusing_count`: coefficient `-0.003830` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002881` (lowers CT win probability)
- `lag_07__CT_defusing_count`: coefficient `-0.002790` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002648` (lowers CT win probability)
- `lag_12__CT_defusing_count`: coefficient `0.002489` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002442` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002416` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `56164`, seconds `57.50`, LSTM delta `-0.4340`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.029150`
- `lag_00__kill_diff_last_3s`: contribution `-0.026624`
- `lag_07__CT_place_FOUNTAIN`: contribution `-0.020110`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.018935`
- `lag_00__T_damage_last_5s`: contribution `-0.013745`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.012069`

### tick `58052`, seconds `87.00`, LSTM delta `+0.3974`

Top all feature movements:
- `lag_15__T4__flash_duration`: contribution `+0.019048`
- `lag_15__T2__flash_duration`: contribution `+0.015470`
- `lag_00__kill_diff_last_3s`: contribution `+0.013312`
- `lag_15__T5__flash_duration`: contribution `+0.012209`
- `lag_11__T1__flash_duration`: contribution `+0.012173`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `+0.019048`
- `lag_15__T2__flash_duration`: contribution `+0.015470`
- `lag_15__T5__flash_duration`: contribution `+0.012209`
- `lag_11__T1__flash_duration`: contribution `+0.012173`
- `lag_03__T2__flash_duration`: contribution `+0.008575`

### tick `59204`, seconds `105.00`, LSTM delta `+0.3212`

Top all feature movements:
- `lag_07__CT_defusing_count`: contribution `+0.027045`
- `lag_12__CT_defusing_count`: contribution `+0.024131`
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.018935`
- `lag_00__T_flash_alpha_mean`: contribution `+0.015561`
- `lag_00__kill_diff_last_3s`: contribution `+0.013312`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015561`

### tick `58980`, seconds `101.50`, LSTM delta `-0.3091`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.089908`
- `lag_05__CT_defusing_count`: contribution `-0.037125`
- `lag_00__T_kills_last_3s`: contribution `-0.014575`
- `lag_00__kill_diff_last_3s`: contribution `-0.013312`
- `lag_01__T_duck_amount_mean`: contribution `-0.009328`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `-0.003395`

### tick `58020`, seconds `86.50`, LSTM delta `-0.2965`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.014575`
- `lag_00__kill_diff_last_3s`: contribution `-0.013312`
- `lag_14__T4__flash_duration`: contribution `-0.009448`
- `lag_00__T_damage_last_5s`: contribution `-0.008772`
- `lag_02__T2__flash_duration`: contribution `-0.008194`

Top utility-only movements:
- `lag_14__T4__flash_duration`: contribution `-0.009448`
- `lag_02__T2__flash_duration`: contribution `-0.008194`
- `lag_14__T1__flash_duration`: contribution `-0.007090`
- `lag_10__T1__flash_duration`: contribution `-0.006162`
- `lag_14__T5__flash_duration`: contribution `-0.006156`
