# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `104038`, seconds `65.00`, LSTM `0.4131`, delta `-0.4792`
- tick `103334`, seconds `54.00`, LSTM `0.5506`, delta `+0.2741`
- tick `103654`, seconds `59.00`, LSTM `0.8884`, delta `+0.1688`
- tick `104070`, seconds `65.50`, LSTM `0.2570`, delta `-0.1561`
- tick `103942`, seconds `63.50`, LSTM `0.9450`, delta `+0.1100`
- tick `104358`, seconds `70.00`, LSTM `0.3629`, delta `+0.0865`
- tick `107142`, seconds `113.50`, LSTM `0.0390`, delta `-0.0806`
- tick `106918`, seconds `110.00`, LSTM `0.0536`, delta `-0.0780`
- tick `104102`, seconds `66.00`, LSTM `0.1821`, delta `-0.0750`
- tick `107046`, seconds `112.00`, LSTM `0.1311`, delta `+0.0744`

## Top 15 local ridge features

- `lag_05__T_flashes_last_5s`: coefficient `-0.004137`, |coef| `0.004137`
- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.003456`, |coef| `0.003456`
- `lag_02__CT_shots_fired_sum`: coefficient `0.003259`, |coef| `0.003259`
- `lag_04__CT_place_UPPERTUNNEL`: coefficient `0.003134`, |coef| `0.003134`
- `lag_00__kill_diff_last_3s`: coefficient `0.002730`, |coef| `0.002730`
- `lag_00__damage_diff_last_5s`: coefficient `0.002239`, |coef| `0.002239`
- `lag_12__CT_place_UPPERTUNNEL`: coefficient `0.002148`, |coef| `0.002148`
- `lag_11__T1__is_scoped`: coefficient `0.002106`, |coef| `0.002106`
- `lag_14__CT_place_UPPERTUNNEL`: coefficient `0.002012`, |coef| `0.002012`
- `lag_00__T_kills_last_3s`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001839`, |coef| `0.001839`
- `lag_09__CT_place_EXTENDEDA`: coefficient `0.001796`, |coef| `0.001796`
- `lag_06__T_flashes_last_5s`: coefficient `-0.001765`, |coef| `0.001765`
- `lag_13__CT_place_UPPERTUNNEL`: coefficient `0.001750`, |coef| `0.001750`
- `lag_01__kill_diff_last_3s`: coefficient `0.001656`, |coef| `0.001656`

## Top 10 utility ridge features

- `lag_05__T_flashes_last_5s`: coefficient `-0.004137` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.001765` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001514` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.001431` (lowers CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.001410` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.001340` (lowers CT win probability)
- `lag_12__T2__utility_total`: coefficient `0.001280` (raises CT win probability)
- `lag_12__T2__flash`: coefficient `0.001143` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.001058` (raises CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.001035` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.003456` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.003259` (raises CT win probability)
- `lag_04__CT_place_UPPERTUNNEL`: coefficient `0.003134` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002730` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002239` (raises CT win probability)
- `lag_12__CT_place_UPPERTUNNEL`: coefficient `0.002148` (raises CT win probability)
- `lag_11__T1__is_scoped`: coefficient `0.002106` (raises CT win probability)
- `lag_14__CT_place_UPPERTUNNEL`: coefficient `0.002012` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001936` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001839` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `104038`, seconds `65.00`, LSTM delta `-0.4792`

Top all feature movements:
- `lag_05__T_flashes_last_5s`: contribution `-0.037482`
- `lag_02__CT_shots_fired_sum`: contribution `-0.031697`
- `lag_04__CT_place_UPPERTUNNEL`: contribution `-0.024037`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `-0.012039`
- `lag_11__T1__is_scoped`: contribution `-0.012032`

Top utility-only movements:
- `lag_05__T_flashes_last_5s`: contribution `-0.037482`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.005203`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.004917`

### tick `103334`, seconds `54.00`, LSTM delta `+0.2741`

Top all feature movements:
- `lag_11__CT_place_UPPERTUNNEL`: contribution `+0.053005`
- `lag_04__CT_place_UPPERTUNNEL`: contribution `+0.024037`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011501`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.007971`
- `lag_00__kill_diff_last_3s`: contribution `+0.006572`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103654`, seconds `59.00`, LSTM delta `+0.1688`

Top all feature movements:
- `lag_14__CT_place_UPPERTUNNEL`: contribution `+0.015428`
- `lag_11__T_place_TUNNELSTAIRS`: contribution `+0.008712`
- `lag_00__kill_diff_last_3s`: contribution `+0.006572`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.006504`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.006107`

Top utility-only movements:
- `lag_15__T_active_infernos`: contribution `+0.002791`

### tick `104070`, seconds `65.50`, LSTM delta `-0.1561`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.015995`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `-0.012039`
- `lag_03__CT_shots_fired_sum`: contribution `+0.009708`
- `lag_04__CT_shots_fired_sum`: contribution `-0.009095`
- `lag_12__T1__is_scoped`: contribution `-0.007248`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.015995`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002519`

### tick `103942`, seconds `63.50`, LSTM delta `+0.1100`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012779`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `-0.012039`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.009849`
- `lag_02__T_flashes_last_5s`: contribution `+0.009382`
- `lag_14__T_place_LOWERTUNNEL`: contribution `+0.006713`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.009382`
