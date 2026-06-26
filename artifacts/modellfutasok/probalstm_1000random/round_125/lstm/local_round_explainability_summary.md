# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `4`

## Largest probability jumps

- tick `28981`, seconds `46.50`, LSTM `0.8749`, delta `+0.3196`
- tick `28181`, seconds `34.00`, LSTM `0.5303`, delta `+0.2842`
- tick `27829`, seconds `28.50`, LSTM `0.2191`, delta `-0.2305`
- tick `28853`, seconds `44.50`, LSTM `0.4308`, delta `-0.1246`
- tick `28213`, seconds `34.50`, LSTM `0.6139`, delta `+0.0836`
- tick `28949`, seconds `46.00`, LSTM `0.5554`, delta `+0.0812`
- tick `28917`, seconds `45.50`, LSTM `0.4741`, delta `+0.0772`
- tick `27861`, seconds `29.00`, LSTM `0.1438`, delta `-0.0754`
- tick `29109`, seconds `48.50`, LSTM `0.9565`, delta `+0.0699`
- tick `28117`, seconds `33.00`, LSTM `0.2102`, delta `+0.0609`

## Top 15 local ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.002731`, |coef| `0.002731`
- `lag_00__kill_diff_last_3s`: coefficient `0.002133`, |coef| `0.002133`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.002024`, |coef| `0.002024`
- `lag_15__CT_place_TUNNEL`: coefficient `-0.001907`, |coef| `0.001907`
- `lag_06__T_place_MAIN`: coefficient `-0.001841`, |coef| `0.001841`
- `lag_04__T_place_MAIN`: coefficient `-0.001808`, |coef| `0.001808`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.001596`, |coef| `0.001596`
- `lag_03__CT_place_HEAVEN`: coefficient `-0.001572`, |coef| `0.001572`
- `lag_00__damage_diff_last_5s`: coefficient `0.001568`, |coef| `0.001568`
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_03__CT_place_BRICKS`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_00__CT_kills_last_3s`: coefficient `0.001459`, |coef| `0.001459`
- `lag_04__CT_place_HEAVEN`: coefficient `-0.001413`, |coef| `0.001413`
- `lag_01__kill_diff_last_3s`: coefficient `0.001360`, |coef| `0.001360`

## Top 10 utility ridge features

- `lag_00__T_A_site_active_infernos`: coefficient `-0.001495` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.001160` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001059` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001046` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000993` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.000956` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000873` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000826` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.000824` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000796` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT2__is_scoped`: coefficient `0.002731` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002133` (raises CT win probability)
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.002024` (lowers CT win probability)
- `lag_15__CT_place_TUNNEL`: coefficient `-0.001907` (lowers CT win probability)
- `lag_06__T_place_MAIN`: coefficient `-0.001841` (lowers CT win probability)
- `lag_04__T_place_MAIN`: coefficient `-0.001808` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001649` (lowers CT win probability)
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `0.001596` (raises CT win probability)
- `lag_03__CT_place_HEAVEN`: coefficient `-0.001572` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001568` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28981`, seconds `46.50`, LSTM delta `+0.3196`

Top all feature movements:
- `lag_15__CT_place_TUNNEL`: contribution `+0.030628`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `+0.028504`
- `lag_15__CT_place_TUNNELSTAIRS`: contribution `+0.016003`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `+0.011732`
- `lag_07__T_bomb_zone_count`: contribution `+0.007593`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.004401`
- `lag_04__T5__flash_duration`: contribution `+0.003973`

### tick `28181`, seconds `34.00`, LSTM delta `+0.2842`

Top all feature movements:
- `lag_09__CT_place_BRICKS`: contribution `+0.025960`
- `lag_00__CT2__is_scoped`: contribution `+0.016716`
- `lag_11__CT_place_FOUNTAIN`: contribution `+0.013408`
- `lag_07__CT_place_BRICKS`: contribution `+0.012173`
- `lag_15__T_place_MAIN`: contribution `+0.008629`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `+0.008477`
- `lag_00__T_A_site_active_infernos`: contribution `+0.004448`

### tick `27829`, seconds `28.50`, LSTM delta `-0.2305`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.016716`
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.012671`
- `lag_06__T_place_MAIN`: contribution `-0.011904`
- `lag_04__T_place_MAIN`: contribution `-0.011689`
- `lag_03__CT_place_HEAVEN`: contribution `-0.008489`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.004448`
- `lag_00__T3__flash_duration`: contribution `-0.002932`
- `lag_03__T_A_site_active_infernos`: contribution `-0.002600`
- `lag_05__T3__flash_duration`: contribution `-0.002434`
- `lag_00__T_active_infernos`: contribution `-0.002178`

### tick `28853`, seconds `44.50`, LSTM delta `-0.1246`

Top all feature movements:
- `lag_15__CT_place_TUNNEL`: contribution `-0.030628`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `-0.028504`
- `lag_15__T_place_MAIN`: contribution `-0.008629`
- `lag_07__CT_place_TUNNELSTAIRS`: contribution `-0.005703`
- `lag_00__kill_diff_last_3s`: contribution `-0.005135`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.004401`

### tick `28213`, seconds `34.50`, LSTM delta `+0.0836`

Top all feature movements:
- `lag_04__T_place_MAIN`: contribution `+0.011689`
- `lag_03__CT_place_HEAVEN`: contribution `-0.008489`
- `lag_10__CT_place_BRICKS`: contribution `+0.006594`
- `lag_01__CT2__is_scoped`: contribution `+0.006443`
- `lag_12__CT_place_FOUNTAIN`: contribution `+0.005383`

Top utility-only movements:
- No utility movement among the top local contributors.
