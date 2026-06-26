# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `19`

## Largest probability jumps

- tick `139812`, seconds `32.50`, LSTM `0.5266`, delta `-0.2536`
- tick `140644`, seconds `45.50`, LSTM `0.7884`, delta `+0.1775`
- tick `139780`, seconds `32.00`, LSTM `0.7802`, delta `+0.1671`
- tick `139044`, seconds `20.50`, LSTM `0.5817`, delta `-0.1644`
- tick `141284`, seconds `55.50`, LSTM `0.9274`, delta `+0.1189`
- tick `140612`, seconds `45.00`, LSTM `0.6109`, delta `+0.0924`
- tick `138404`, seconds `10.50`, LSTM `0.8015`, delta `+0.0860`
- tick `141636`, seconds `61.00`, LSTM `0.8998`, delta `-0.0526`
- tick `139940`, seconds `34.50`, LSTM `0.4099`, delta `-0.0520`
- tick `140004`, seconds `35.50`, LSTM `0.4410`, delta `+0.0454`

## Top 15 local ridge features

- `lag_11__T_place_SNIPERSNEST`: coefficient `0.002527`, |coef| `0.002527`
- `lag_00__kill_diff_last_3s`: coefficient `0.002355`, |coef| `0.002355`
- `lag_10__CT_place_TRAMP`: coefficient `-0.002123`, |coef| `0.002123`
- `lag_00__damage_diff_last_5s`: coefficient `0.002086`, |coef| `0.002086`
- `lag_15__T_place_SNIPERSNEST`: coefficient `-0.002072`, |coef| `0.002072`
- `lag_10__CT_place_PALACEALLEY`: coefficient `0.002044`, |coef| `0.002044`
- `lag_10__T_place_SNIPERSNEST`: coefficient `-0.001926`, |coef| `0.001926`
- `lag_15__CT_place_TRAMP`: coefficient `0.001840`, |coef| `0.001840`
- `lag_07__T_place_JUNGLE`: coefficient `0.001819`, |coef| `0.001819`
- `lag_11__CT2__is_scoped`: coefficient `0.001797`, |coef| `0.001797`
- `lag_11__T5__duck_amount`: coefficient `0.001603`, |coef| `0.001603`
- `lag_11__T_place_STAIRS`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_07__T_place_CTSPAWN`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_00__CT_kills_last_3s`: coefficient `0.001514`, |coef| `0.001514`
- `lag_10__CT_place_TRUCK`: coefficient `-0.001444`, |coef| `0.001444`

## Top 10 utility ridge features

- `lag_02__T1__flash_duration`: coefficient `0.001195` (raises CT win probability)
- `lag_06__T_smokes_last_5s`: coefficient `-0.001018` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000986` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.000898` (raises CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.000714` (lowers CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000709` (raises CT win probability)
- `lag_12__T_smokes_last_5s`: coefficient `0.000697` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000690` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.000651` (raises CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `-0.000638` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_SNIPERSNEST`: coefficient `0.002527` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002355` (raises CT win probability)
- `lag_10__CT_place_TRAMP`: coefficient `-0.002123` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002086` (raises CT win probability)
- `lag_15__T_place_SNIPERSNEST`: coefficient `-0.002072` (lowers CT win probability)
- `lag_10__CT_place_PALACEALLEY`: coefficient `0.002044` (raises CT win probability)
- `lag_10__T_place_SNIPERSNEST`: coefficient `-0.001926` (lowers CT win probability)
- `lag_15__CT_place_TRAMP`: coefficient `0.001840` (raises CT win probability)
- `lag_07__T_place_JUNGLE`: coefficient `0.001819` (raises CT win probability)
- `lag_11__CT2__is_scoped`: coefficient `0.001797` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `139812`, seconds `32.50`, LSTM delta `-0.2536`

Top all feature movements:
- `lag_11__T_place_SNIPERSNEST`: contribution `-0.044907`
- `lag_15__T_place_SNIPERSNEST`: contribution `-0.036817`
- `lag_11__CT2__is_scoped`: contribution `-0.011001`
- `lag_10__CT_place_TRUCK`: contribution `-0.009317`
- `lag_11__T_place_CTSPAWN`: contribution `-0.006763`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140644`, seconds `45.50`, LSTM delta `+0.1775`

Top all feature movements:
- `lag_10__CT_place_PALACEALLEY`: contribution `+0.031205`
- `lag_10__CT_place_TRAMP`: contribution `+0.028605`
- `lag_15__CT_place_TRAMP`: contribution `+0.024782`
- `lag_07__T_place_JUNGLE`: contribution `+0.023567`
- `lag_07__T_place_CTSPAWN`: contribution `+0.007461`

Top utility-only movements:
- `lag_00__T4__smoke`: contribution `+0.001275`

### tick `139780`, seconds `32.00`, LSTM delta `+0.1671`

Top all feature movements:
- `lag_10__T_place_SNIPERSNEST`: contribution `+0.034224`
- `lag_14__T_place_SNIPERSNEST`: contribution `+0.021731`
- `lag_14__CT_place_TRUCK`: contribution `+0.008793`
- `lag_11__T5__duck_amount`: contribution `+0.006086`
- `lag_00__kill_diff_last_3s`: contribution `+0.005667`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139044`, seconds `20.50`, LSTM delta `-0.1644`

Top all feature movements:
- `lag_11__CT2__is_scoped`: contribution `-0.011001`
- `lag_02__T1__flash_duration`: contribution `-0.006552`
- `lag_13__CT_place_SNIPERSNEST`: contribution `-0.006004`
- `lag_09__CT2__is_scoped`: contribution `-0.005980`
- `lag_00__kill_diff_last_3s`: contribution `-0.005667`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.006552`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.002453`
- `lag_01__CT_B_site_active_infernos`: contribution `-0.002434`

### tick `141284`, seconds `55.50`, LSTM delta `+0.1189`

Top all feature movements:
- `lag_11__T_place_STAIRS`: contribution `+0.030510`
- `lag_13__T_place_STAIRS`: contribution `+0.021617`
- `lag_00__kill_diff_last_3s`: contribution `+0.005667`
- `lag_00__CT_kills_last_3s`: contribution `+0.004370`
- `lag_00__damage_diff_last_5s`: contribution `+0.004189`

Top utility-only movements:
- No utility movement among the top local contributors.
