# Local Round Explainability

- csv_path: `processed_full\esl_pro_league_season_21\esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY\vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `12`

## Largest probability jumps

- tick `88645`, seconds `14.50`, LSTM `0.0880`, delta `-0.3219`
- tick `89125`, seconds `22.00`, LSTM `0.5693`, delta `+0.2836`
- tick `92741`, seconds `78.50`, LSTM `0.2144`, delta `+0.1641`
- tick `90213`, seconds `39.00`, LSTM `0.3346`, delta `-0.1576`
- tick `88613`, seconds `14.00`, LSTM `0.4099`, delta `+0.1464`
- tick `90181`, seconds `38.50`, LSTM `0.4922`, delta `-0.1345`
- tick `88677`, seconds `15.00`, LSTM `0.2116`, delta `+0.1236`
- tick `90245`, seconds `39.50`, LSTM `0.2190`, delta `-0.1156`
- tick `92037`, seconds `67.50`, LSTM `0.0487`, delta `-0.0910`
- tick `88549`, seconds `13.00`, LSTM `0.2563`, delta `-0.0604`

## Top 15 local ridge features

- `lag_10__CT_place_OUTSIDELONG`: coefficient `-0.002787`, |coef| `0.002787`
- `lag_00__kill_diff_last_3s`: coefficient `0.002754`, |coef| `0.002754`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002422`, |coef| `0.002422`
- `lag_12__CT_place_OUTSIDELONG`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_00__damage_diff_last_5s`: coefficient `0.002193`, |coef| `0.002193`
- `lag_00__T_kills_last_3s`: coefficient `-0.002151`, |coef| `0.002151`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `-0.001995`, |coef| `0.001995`
- `lag_10__CT_place_RUINS`: coefficient `0.001901`, |coef| `0.001901`
- `lag_04__CT_place_BRIDGE`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_11__T_place_STREET`: coefficient `0.001884`, |coef| `0.001884`
- `lag_09__CT_place_BRIDGE`: coefficient `0.001833`, |coef| `0.001833`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001759`, |coef| `0.001759`
- `lag_01__T_shots_fired_sum`: coefficient `0.001745`, |coef| `0.001745`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `0.001744`, |coef| `0.001744`
- `lag_01__CT_place_RUINS`: coefficient `0.001596`, |coef| `0.001596`

## Top 10 utility ridge features

- `lag_01__CT3__flash`: coefficient `0.001059` (raises CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.001035` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000995` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `0.000943` (raises CT win probability)
- `lag_05__CT3__molly`: coefficient `-0.000943` (lowers CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.000941` (raises CT win probability)
- `lag_12__CT1__molly`: coefficient `-0.000923` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000895` (raises CT win probability)
- `lag_01__CT3__utility_total`: coefficient `0.000864` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000858` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_OUTSIDELONG`: coefficient `-0.002787` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002754` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002422` (raises CT win probability)
- `lag_12__CT_place_OUTSIDELONG`: coefficient `-0.002268` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002193` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002151` (lowers CT win probability)
- `lag_09__CT_place_OUTSIDELONG`: coefficient `-0.001995` (lowers CT win probability)
- `lag_10__CT_place_RUINS`: coefficient `0.001901` (raises CT win probability)
- `lag_04__CT_place_BRIDGE`: coefficient `-0.001899` (lowers CT win probability)
- `lag_11__T_place_STREET`: coefficient `0.001884` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `88645`, seconds `14.50`, LSTM delta `-0.3219`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.021871`
- `lag_04__CT_place_BRIDGE`: contribution `-0.021764`
- `lag_09__CT_place_BRIDGE`: contribution `-0.021009`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.017840`
- `lag_00__T_kills_last_3s`: contribution `-0.013628`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `-0.004739`
- `lag_12__CT2__flash_duration`: contribution `-0.004461`
- `lag_12__CT_flash_duration_sum`: contribution `-0.004310`

### tick `89125`, seconds `22.00`, LSTM delta `+0.2836`

Top all feature movements:
- `lag_12__CT_place_OUTSIDELONG`: contribution `+0.022999`
- `lag_04__CT_place_BRIDGE`: contribution `+0.021764`
- `lag_11__CT_place_OUTSIDELONG`: contribution `-0.014863`
- `lag_15__CT_shots_fired_sum`: contribution `+0.011517`
- `lag_11__T_place_STREET`: contribution `+0.010359`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `+0.005372`
- `lag_15__CT2__flash_duration`: contribution `+0.005057`

### tick `92741`, seconds `78.50`, LSTM delta `+0.1641`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.011956`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008412`
- `lag_01__CT_duck_amount_mean`: contribution `+0.007438`
- `lag_00__kill_diff_last_3s`: contribution `+0.006629`
- `lag_01__T_shots_fired_sum`: contribution `+0.006540`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `+0.003420`
- `lag_12__CT1__molly`: contribution `+0.002297`
- `lag_00__T1__flash`: contribution `+0.002105`
- `lag_03__T_B_site_active_infernos`: contribution `+0.001992`

### tick `90213`, seconds `39.00`, LSTM delta `-0.1576`

Top all feature movements:
- `lag_10__CT_place_OUTSIDELONG`: contribution `-0.028268`
- `lag_01__CT_place_OUTSIDELONG`: contribution `-0.007045`
- `lag_10__CT_place_RUINS`: contribution `-0.006641`
- `lag_12__CT2__duck_amount`: contribution `-0.004010`
- `lag_05__CT2__duck_amount`: contribution `-0.003978`

Top utility-only movements:
- `lag_05__CT3__molly`: contribution `-0.002327`

### tick `88613`, seconds `14.00`, LSTM delta `+0.1464`

Top all feature movements:
- `lag_02__CT_place_OUTSIDELONG`: contribution `+0.017693`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015141`
- `lag_03__CT_place_BRIDGE`: contribution `-0.012003`
- `lag_08__CT_place_BRIDGE`: contribution `+0.009512`
- `lag_11__T_place_TSTAIRS`: contribution `+0.008784`

Top utility-only movements:
- No utility movement among the top local contributors.
