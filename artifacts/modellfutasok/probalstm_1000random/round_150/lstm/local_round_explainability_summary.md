# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `20`

## Largest probability jumps

- tick `144683`, seconds `23.50`, LSTM `0.5681`, delta `-0.1637`
- tick `149099`, seconds `92.50`, LSTM `0.6422`, delta `-0.1534`
- tick `144939`, seconds `27.50`, LSTM `0.7531`, delta `+0.1525`
- tick `145195`, seconds `31.50`, LSTM `0.7476`, delta `-0.1228`
- tick `145099`, seconds `30.00`, LSTM `0.7631`, delta `-0.1158`
- tick `145131`, seconds `30.50`, LSTM `0.8767`, delta `+0.1136`
- tick `145067`, seconds `29.50`, LSTM `0.8789`, delta `+0.0946`
- tick `144779`, seconds `25.00`, LSTM `0.6532`, delta `+0.0914`
- tick `144331`, seconds `18.00`, LSTM `0.7260`, delta `+0.0744`
- tick `145259`, seconds `32.50`, LSTM `0.6987`, delta `-0.0710`

## Top 15 local ridge features

- `lag_01__CT_place_HUT`: coefficient `-0.002485`, |coef| `0.002485`
- `lag_14__CT_place_LOBBY`: coefficient `-0.002086`, |coef| `0.002086`
- `lag_14__CT_place_VENDING`: coefficient `0.001942`, |coef| `0.001942`
- `lag_04__T_place_MINI`: coefficient `0.001640`, |coef| `0.001640`
- `lag_00__kill_diff_last_3s`: coefficient `0.001449`, |coef| `0.001449`
- `lag_00__damage_diff_last_5s`: coefficient `0.001440`, |coef| `0.001440`
- `lag_15__T_place_MINI`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_14__T_place_MINI`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_13__T_place_MINI`: coefficient `-0.001266`, |coef| `0.001266`
- `lag_00__T_kills_last_3s`: coefficient `-0.001252`, |coef| `0.001252`
- `lag_05__CT_place_HUT`: coefficient `-0.001203`, |coef| `0.001203`
- `lag_01__CT_place_LOBBY`: coefficient `0.001193`, |coef| `0.001193`
- `lag_00__CT_place_HUT`: coefficient `-0.001165`, |coef| `0.001165`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001133`, |coef| `0.001133`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001102`, |coef| `0.001102`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001052` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000863` (raises CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.000832` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000816` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000800` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.000739` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000699` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000682` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.000649` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000532` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_HUT`: coefficient `-0.002485` (lowers CT win probability)
- `lag_14__CT_place_LOBBY`: coefficient `-0.002086` (lowers CT win probability)
- `lag_14__CT_place_VENDING`: coefficient `0.001942` (raises CT win probability)
- `lag_04__T_place_MINI`: coefficient `0.001640` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001449` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001440` (raises CT win probability)
- `lag_15__T_place_MINI`: coefficient `-0.001273` (lowers CT win probability)
- `lag_14__T_place_MINI`: coefficient `-0.001273` (lowers CT win probability)
- `lag_13__T_place_MINI`: coefficient `-0.001266` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001252` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `144683`, seconds `23.50`, LSTM delta `-0.1637`

Top all feature movements:
- `lag_11__CT_place_SECRET`: contribution `-0.010409`
- `lag_12__CT_place_HUTROOF`: contribution `-0.006404`
- `lag_12__T3__flash_duration`: contribution `-0.006297`
- `lag_13__T_place_HUT`: contribution `-0.005913`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.005757`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `-0.006297`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.005757`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.005495`
- `lag_11__CT4__flash_duration`: contribution `-0.004522`
- `lag_04__T4__flash_duration`: contribution `-0.003372`

### tick `149099`, seconds `92.50`, LSTM delta `-0.1534`

Top all feature movements:
- `lag_14__CT_place_VENDING`: contribution `-0.033284`
- `lag_01__CT_place_HUT`: contribution `-0.024234`
- `lag_04__T_place_MINI`: contribution `-0.022821`
- `lag_14__CT_place_LOBBY`: contribution `-0.017076`
- `lag_01__CT_place_LOBBY`: contribution `-0.009763`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144939`, seconds `27.50`, LSTM delta `+0.1525`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.011228`
- `lag_02__T_place_HUT`: contribution `+0.008162`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007558`
- `lag_07__T_shots_fired_sum`: contribution `+0.007109`
- `lag_01__T_place_HUT`: contribution `+0.006817`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.011228`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007558`
- `lag_05__CT4__flash_duration`: contribution `+0.005542`
- `lag_06__T3__flash_duration`: contribution `+0.002776`

### tick `145195`, seconds `31.50`, LSTM delta `-0.1228`

Top all feature movements:
- `lag_02__T_place_HUT`: contribution `-0.008162`
- `lag_05__T_place_HUT`: contribution `-0.006497`
- `lag_09__T_place_HUT`: contribution `-0.006130`
- `lag_13__CT4__flash_duration`: contribution `-0.005864`
- `lag_03__CT_place_RAFTERS`: contribution `-0.005160`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `-0.005864`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.004556`
- `lag_08__CT4__flash_duration`: contribution `-0.003522`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.003073`
- `lag_14__T3__flash_duration`: contribution `-0.002967`

### tick `145099`, seconds `30.00`, LSTM delta `-0.1158`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.008263`
- `lag_02__T_place_HUT`: contribution `+0.008162`
- `lag_07__T_place_HUT`: contribution `-0.007664`
- `lag_06__T_place_HUT`: contribution `+0.005694`
- `lag_11__CT2__is_scoped`: contribution `-0.005286`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `-0.005181`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.004626`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.003050`
- `lag_11__T3__flash_duration`: contribution `-0.002951`
