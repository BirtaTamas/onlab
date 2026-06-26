# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `17`

## Largest probability jumps

- tick `122877`, seconds `27.00`, LSTM `0.8260`, delta `+0.3129`
- tick `122845`, seconds `26.50`, LSTM `0.5132`, delta `-0.2708`
- tick `123581`, seconds `38.00`, LSTM `0.8823`, delta `+0.2115`
- tick `125629`, seconds `70.00`, LSTM `0.9400`, delta `+0.2094`
- tick `122589`, seconds `22.50`, LSTM `0.6546`, delta `-0.1393`
- tick `122365`, seconds `19.00`, LSTM `0.8241`, delta `+0.1172`
- tick `122333`, seconds `18.50`, LSTM `0.7068`, delta `+0.1164`
- tick `125117`, seconds `62.00`, LSTM `0.7381`, delta `-0.0793`
- tick `123101`, seconds `30.50`, LSTM `0.6874`, delta `-0.0728`
- tick `125085`, seconds `61.50`, LSTM `0.8174`, delta `-0.0679`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002517`, |coef| `0.002517`
- `lag_05__CT_place_HUT`: coefficient `-0.002474`, |coef| `0.002474`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002467`, |coef| `0.002467`
- `lag_09__CT2__is_scoped`: coefficient `-0.002343`, |coef| `0.002343`
- `lag_15__CT_place_TROPHY`: coefficient `-0.002133`, |coef| `0.002133`
- `lag_12__CT_place_CONTROL`: coefficient `-0.002054`, |coef| `0.002054`
- `lag_10__T_place_GARAGE`: coefficient `0.001958`, |coef| `0.001958`
- `lag_11__CT_place_CRANE`: coefficient `0.001957`, |coef| `0.001957`
- `lag_00__CT_kills_last_3s`: coefficient `0.001902`, |coef| `0.001902`
- `lag_15__CT_place_HUT`: coefficient `-0.001853`, |coef| `0.001853`
- `lag_00__T5__shots_fired`: coefficient `-0.001749`, |coef| `0.001749`
- `lag_15__CT_place_CONTROL`: coefficient `0.001698`, |coef| `0.001698`
- `lag_12__CT_place_CRANE`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_00__CT_place_CRANE`: coefficient `0.001645`, |coef| `0.001645`
- `lag_09__CT_place_VENTS`: coefficient `0.001641`, |coef| `0.001641`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001545` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000940` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.000924` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `-0.000872` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000773` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000756` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.000749` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.000737` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `0.000693` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `0.000675` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002517` (raises CT win probability)
- `lag_05__CT_place_HUT`: coefficient `-0.002474` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002467` (lowers CT win probability)
- `lag_09__CT2__is_scoped`: coefficient `-0.002343` (lowers CT win probability)
- `lag_15__CT_place_TROPHY`: coefficient `-0.002133` (lowers CT win probability)
- `lag_12__CT_place_CONTROL`: coefficient `-0.002054` (lowers CT win probability)
- `lag_10__T_place_GARAGE`: coefficient `0.001958` (raises CT win probability)
- `lag_11__CT_place_CRANE`: coefficient `0.001957` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001902` (raises CT win probability)
- `lag_15__CT_place_HUT`: coefficient `-0.001853` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `122877`, seconds `27.00`, LSTM delta `+0.3129`

Top all feature movements:
- `lag_12__CT_place_CRANE`: contribution `+0.027047`
- `lag_02__CT_place_CRANE`: contribution `+0.024143`
- `lag_00__T_shots_fired_sum`: contribution `+0.024045`
- `lag_09__CT2__is_scoped`: contribution `+0.014342`
- `lag_00__T5__shots_fired`: contribution `+0.013981`

Top utility-only movements:
- `lag_11__CT2__flash_duration`: contribution `+0.004155`
- `lag_08__CT4__flash_duration`: contribution `+0.003521`

### tick `122845`, seconds `26.50`, LSTM delta `-0.2708`

Top all feature movements:
- `lag_11__CT_place_CRANE`: contribution `-0.032102`
- `lag_14__CT_place_CRANE`: contribution `-0.017241`
- `lag_01__CT_place_CRANE`: contribution `-0.015809`
- `lag_09__CT2__is_scoped`: contribution `-0.014342`
- `lag_08__T_shots_fired_sum`: contribution `-0.010354`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123581`, seconds `38.00`, LSTM delta `+0.2115`

Top all feature movements:
- `lag_05__CT_place_HUT`: contribution `+0.024124`
- `lag_10__T_place_GARAGE`: contribution `+0.023542`
- `lag_00__T_place_GARAGE`: contribution `+0.017900`
- `lag_05__CT_place_LOBBY`: contribution `+0.010925`
- `lag_11__CT_place_HUT`: contribution `+0.009841`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.004989`
- `lag_04__T4__flash_duration`: contribution `+0.004906`
- `lag_05__T5__flash_duration`: contribution `+0.003079`
- `lag_13__T5__flash_duration`: contribution `+0.002883`

### tick `125629`, seconds `70.00`, LSTM delta `+0.2094`

Top all feature movements:
- `lag_15__CT_place_TROPHY`: contribution `+0.031500`
- `lag_12__CT_place_CONTROL`: contribution `+0.021323`
- `lag_15__CT_place_HUT`: contribution `+0.018071`
- `lag_15__CT_place_CONTROL`: contribution `+0.017626`
- `lag_09__CT_place_VENTS`: contribution `+0.013766`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.009374`
- `lag_06__T_A_site_active_infernos`: contribution `+0.001927`

### tick `122589`, seconds `22.50`, LSTM delta `-0.1393`

Top all feature movements:
- `lag_06__CT_place_CRANE`: contribution `-0.019081`
- `lag_00__T_shots_fired_sum`: contribution `-0.016647`
- `lag_03__CT_place_CRANE`: contribution `-0.007507`
- `lag_00__kill_diff_last_3s`: contribution `-0.006059`
- `lag_00__T5__shots_fired`: contribution `-0.005377`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.004671`
- `lag_02__CT2__flash_duration`: contribution `-0.003682`
- `lag_10__T3__flash_duration`: contribution `-0.002328`
