# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `11`

## Largest probability jumps

- tick `83843`, seconds `92.50`, LSTM `0.0889`, delta `-0.1951`
- tick `79971`, seconds `32.00`, LSTM `0.7478`, delta `+0.1787`
- tick `83523`, seconds `87.50`, LSTM `0.5598`, delta `-0.1669`
- tick `83619`, seconds `89.00`, LSTM `0.4772`, delta `-0.1630`
- tick `83651`, seconds `89.50`, LSTM `0.3289`, delta `-0.1483`
- tick `83587`, seconds `88.50`, LSTM `0.6402`, delta `+0.1127`
- tick `79555`, seconds `25.50`, LSTM `0.5758`, delta `+0.0864`
- tick `83491`, seconds `87.00`, LSTM `0.7268`, delta `+0.0474`
- tick `82083`, seconds `65.00`, LSTM `0.5957`, delta `-0.0469`
- tick `82051`, seconds `64.50`, LSTM `0.6426`, delta `+0.0430`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003568`, |coef| `0.003568`
- `lag_00__T_kills_last_3s`: coefficient `-0.003280`, |coef| `0.003280`
- `lag_00__CT_place_TUNNELS`: coefficient `0.002953`, |coef| `0.002953`
- `lag_00__damage_diff_last_5s`: coefficient `0.002941`, |coef| `0.002941`
- `lag_00__T_damage_last_5s`: coefficient `-0.002270`, |coef| `0.002270`
- `lag_14__CT_place_HUT`: coefficient `0.002212`, |coef| `0.002212`
- `lag_05__CT_flashes_last_5s`: coefficient `-0.002093`, |coef| `0.002093`
- `lag_08__CT3__is_walking`: coefficient `0.002044`, |coef| `0.002044`
- `lag_08__CT_walking_count`: coefficient `0.001809`, |coef| `0.001809`
- `lag_13__CT_place_VENTS`: coefficient `0.001805`, |coef| `0.001805`
- `lag_01__CT5__is_scoped`: coefficient `0.001762`, |coef| `0.001762`
- `lag_05__CT_flashed_players`: coefficient `-0.001753`, |coef| `0.001753`
- `lag_03__CT_flashed_players`: coefficient `0.001723`, |coef| `0.001723`
- `lag_08__CT1__is_walking`: coefficient `0.001672`, |coef| `0.001672`
- `lag_02__CT_place_DECON`: coefficient `0.001659`, |coef| `0.001659`

## Top 10 utility ridge features

- `lag_05__CT_flashes_last_5s`: coefficient `-0.002093` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001614` (raises CT win probability)
- `lag_01__CT5__flash`: coefficient `0.001358` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001177` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.001174` (lowers CT win probability)
- `lag_04__CT3__smoke`: coefficient `0.001043` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.000935` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000907` (raises CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `-0.000896` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000891` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003568` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003280` (lowers CT win probability)
- `lag_00__CT_place_TUNNELS`: coefficient `0.002953` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002941` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002270` (lowers CT win probability)
- `lag_14__CT_place_HUT`: coefficient `0.002212` (raises CT win probability)
- `lag_08__CT3__is_walking`: coefficient `0.002044` (raises CT win probability)
- `lag_08__CT_walking_count`: coefficient `0.001809` (raises CT win probability)
- `lag_13__CT_place_VENTS`: coefficient `0.001805` (raises CT win probability)
- `lag_01__CT5__is_scoped`: coefficient `0.001762` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `83843`, seconds `92.50`, LSTM delta `-0.1951`

Top all feature movements:
- `lag_02__CT_place_DECON`: contribution `-0.026384`
- `lag_05__CT_flashes_last_5s`: contribution `-0.023016`
- `lag_00__T_kills_last_3s`: contribution `-0.010392`
- `lag_00__CT_place_TUNNELS`: contribution `-0.009037`
- `lag_00__kill_diff_last_3s`: contribution `-0.008587`

Top utility-only movements:
- `lag_05__CT_flashes_last_5s`: contribution `-0.023016`
- `lag_07__CT5__flash`: contribution `-0.002394`

### tick `79971`, seconds `32.00`, LSTM delta `+0.1787`

Top all feature movements:
- `lag_14__CT_place_HUT`: contribution `+0.021573`
- `lag_00__kill_diff_last_3s`: contribution `+0.008587`
- `lag_05__T_place_TROPHY`: contribution `+0.007036`
- `lag_07__CT4__flash_duration`: contribution `+0.006684`
- `lag_00__damage_diff_last_5s`: contribution `+0.006634`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.006684`

### tick `83523`, seconds `87.50`, LSTM delta `-0.1669`

Top all feature movements:
- `lag_13__CT_place_VENTS`: contribution `-0.015148`
- `lag_00__T_kills_last_3s`: contribution `-0.010392`
- `lag_00__CT_place_TUNNELS`: contribution `-0.009037`
- `lag_00__kill_diff_last_3s`: contribution `-0.008587`
- `lag_01__CT_flashed_players`: contribution `-0.007604`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83619`, seconds `89.00`, LSTM delta `-0.1630`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.010392`
- `lag_00__CT_place_TUNNELS`: contribution `-0.009037`
- `lag_00__kill_diff_last_3s`: contribution `-0.008587`
- `lag_04__CT_flashed_players`: contribution `-0.006644`
- `lag_01__CT5__is_scoped`: contribution `-0.006302`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.005730`

### tick `83651`, seconds `89.50`, LSTM delta `-0.1483`

Top all feature movements:
- `lag_05__CT_flashed_players`: contribution `-0.011516`
- `lag_01__T_shots_fired_sum`: contribution `-0.006096`
- `lag_01__CT5__flash`: contribution `-0.004821`
- `lag_01__CT_place_TUNNELS`: contribution `-0.004247`
- `lag_00__damage_diff_last_5s`: contribution `-0.003914`

Top utility-only movements:
- `lag_01__CT5__flash`: contribution `-0.004821`
