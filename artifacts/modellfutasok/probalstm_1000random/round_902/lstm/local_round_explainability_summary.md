# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `25333`, seconds `79.00`, LSTM `0.3062`, delta `-0.2271`
- tick `26037`, seconds `90.00`, LSTM `0.0655`, delta `-0.2152`
- tick `21813`, seconds `24.00`, LSTM `0.6420`, delta `+0.1176`
- tick `23285`, seconds `47.00`, LSTM `0.5243`, delta `-0.1114`
- tick `24341`, seconds `63.50`, LSTM `0.3892`, delta `-0.0983`
- tick `25365`, seconds `79.50`, LSTM `0.2140`, delta `-0.0922`
- tick `25269`, seconds `78.00`, LSTM `0.5153`, delta `+0.0883`
- tick `24373`, seconds `64.00`, LSTM `0.4547`, delta `+0.0656`
- tick `25845`, seconds `87.00`, LSTM `0.2633`, delta `+0.0585`
- tick `24757`, seconds `70.00`, LSTM `0.4593`, delta `+0.0543`

## Top 15 local ridge features

- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.002168`, |coef| `0.002168`
- `lag_00__T_kills_last_3s`: coefficient `-0.001960`, |coef| `0.001960`
- `lag_00__kill_diff_last_3s`: coefficient `0.001842`, |coef| `0.001842`
- `lag_03__CT_place_VENDING`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_09__CT_place_HELL`: coefficient `0.001629`, |coef| `0.001629`
- `lag_09__T_place_VENDING`: coefficient `0.001532`, |coef| `0.001532`
- `lag_03__CT_place_TROPHY`: coefficient `0.001487`, |coef| `0.001487`
- `lag_13__CT2__is_walking`: coefficient `-0.001464`, |coef| `0.001464`
- `lag_13__CT_place_HEAVEN`: coefficient `0.001380`, |coef| `0.001380`
- `lag_00__damage_diff_last_5s`: coefficient `0.001354`, |coef| `0.001354`
- `lag_03__CT5__is_walking`: coefficient `0.001314`, |coef| `0.001314`
- `lag_00__T_damage_last_5s`: coefficient `-0.001253`, |coef| `0.001253`
- `lag_13__CT_place_HELL`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_07__CT_B_site_active_smokes`: coefficient `0.001222`, |coef| `0.001222`
- `lag_06__CT_place_TROPHY`: coefficient `-0.001220`, |coef| `0.001220`

## Top 10 utility ridge features

- `lag_07__CT_B_site_active_smokes`: coefficient `0.001222` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `0.001128` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `0.000945` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.000927` (lowers CT win probability)
- `lag_07__CT_active_smokes`: coefficient `0.000876` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000873` (lowers CT win probability)
- `lag_09__T2__smoke`: coefficient `0.000761` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000752` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000740` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.000730` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.002168` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001960` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001842` (raises CT win probability)
- `lag_03__CT_place_VENDING`: coefficient `-0.001700` (lowers CT win probability)
- `lag_09__CT_place_HELL`: coefficient `0.001629` (raises CT win probability)
- `lag_09__T_place_VENDING`: coefficient `0.001532` (raises CT win probability)
- `lag_03__CT_place_TROPHY`: coefficient `0.001487` (raises CT win probability)
- `lag_13__CT2__is_walking`: coefficient `-0.001464` (lowers CT win probability)
- `lag_13__CT_place_HEAVEN`: coefficient `0.001380` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001354` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25333`, seconds `79.00`, LSTM delta `-0.2271`

Top all feature movements:
- `lag_09__CT_place_LOCKERROOM`: contribution `-0.026983`
- `lag_09__CT_place_HELL`: contribution `-0.008833`
- `lag_09__T_place_VENDING`: contribution `-0.007768`
- `lag_13__CT_place_HEAVEN`: contribution `-0.007453`
- `lag_13__CT_place_HELL`: contribution `-0.006727`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26037`, seconds `90.00`, LSTM delta `-0.2152`

Top all feature movements:
- `lag_03__CT_place_VENDING`: contribution `-0.029137`
- `lag_03__CT_place_TROPHY`: contribution `-0.021966`
- `lag_06__CT_place_TROPHY`: contribution `-0.018020`
- `lag_07__T_place_MINI`: contribution `-0.011658`
- `lag_08__CT_place_HUT`: contribution `-0.011466`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21813`, seconds `24.00`, LSTM delta `+0.1176`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `+0.012091`
- `lag_03__T_shots_fired_sum`: contribution `+0.009694`
- `lag_03__T3__shots_fired`: contribution `+0.008517`
- `lag_15__CT_place_CONTROL`: contribution `+0.006519`
- `lag_15__T5__flash_duration`: contribution `+0.005764`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `+0.005764`
- `lag_15__CT2__flash_duration`: contribution `+0.005406`

### tick `23285`, seconds `47.00`, LSTM delta `-0.1114`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006210`
- `lag_08__CT_place_MINI`: contribution `-0.005655`
- `lag_00__CT_place_RAFTERS`: contribution `-0.005007`
- `lag_00__kill_diff_last_3s`: contribution `-0.004434`
- `lag_01__CT_place_MINI`: contribution `-0.004345`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `-0.002203`

### tick `24341`, seconds `63.50`, LSTM delta `-0.0983`

Top all feature movements:
- `lag_03__CT_place_SQUEAKY`: contribution `-0.013986`
- `lag_05__CT_place_RAFTERS`: contribution `-0.005411`
- `lag_05__CT_place_HEAVEN`: contribution `-0.004314`
- `lag_08__T_place_VENDING`: contribution `-0.003630`
- `lag_00__CT_place_SQUEAKY`: contribution `+0.002936`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `-0.002500`
- `lag_07__CT_B_site_active_smokes`: contribution `-0.002029`
- `lag_05__T_utility_damage_last_5s`: contribution `-0.001843`
- `lag_07__CT_A_site_active_smokes`: contribution `-0.001816`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.001451`
