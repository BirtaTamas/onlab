# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `113999`, seconds `28.00`, LSTM `0.4452`, delta `+0.2616`
- tick `114671`, seconds `38.50`, LSTM `0.7925`, delta `+0.2300`
- tick `115119`, seconds `45.50`, LSTM `0.6833`, delta `-0.1449`
- tick `113935`, seconds `27.00`, LSTM `0.1598`, delta `+0.1316`
- tick `114127`, seconds `30.00`, LSTM `0.4115`, delta `-0.1190`
- tick `114543`, seconds `36.50`, LSTM `0.6149`, delta `+0.0992`
- tick `113359`, seconds `18.00`, LSTM `0.3146`, delta `-0.0710`
- tick `114607`, seconds `37.50`, LSTM `0.5485`, delta `-0.0672`
- tick `113423`, seconds `19.00`, LSTM `0.1912`, delta `-0.0668`
- tick `113327`, seconds `17.50`, LSTM `0.3857`, delta `-0.0649`

## Top 15 local ridge features

- `lag_14__T_place_OBSERVATION`: coefficient `0.003099`, |coef| `0.003099`
- `lag_07__T_place_OBSERVATION`: coefficient `-0.002747`, |coef| `0.002747`
- `lag_05__CT_place_DECON`: coefficient `0.002683`, |coef| `0.002683`
- `lag_04__T_place_RAMP`: coefficient `-0.002443`, |coef| `0.002443`
- `lag_07__CT_place_VENTS`: coefficient `-0.002258`, |coef| `0.002258`
- `lag_00__kill_diff_last_3s`: coefficient `0.002188`, |coef| `0.002188`
- `lag_03__CT_place_SECRET`: coefficient `-0.002127`, |coef| `0.002127`
- `lag_03__T_place_RAMP`: coefficient `-0.001980`, |coef| `0.001980`
- `lag_00__damage_diff_last_5s`: coefficient `0.001775`, |coef| `0.001775`
- `lag_00__T_duck_amount_mean`: coefficient `0.001742`, |coef| `0.001742`
- `lag_14__CT_place_DECON`: coefficient `-0.001704`, |coef| `0.001704`
- `lag_15__T_place_CONTROL`: coefficient `-0.001675`, |coef| `0.001675`
- `lag_09__CT_place_DECON`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_00__CT_kills_last_3s`: coefficient `0.001663`, |coef| `0.001663`
- `lag_15__T4__is_walking`: coefficient `0.001603`, |coef| `0.001603`

## Top 10 utility ridge features

- `lag_01__T_utility_inv`: coefficient `-0.000050` (lowers CT win probability)
- `lag_01__CT_flash_inv`: coefficient `-0.000049` (lowers CT win probability)
- `lag_01__T_flash_inv`: coefficient `-0.000048` (lowers CT win probability)
- `lag_01__CT_utility_inv`: coefficient `-0.000048` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000043` (lowers CT win probability)
- `lag_01__CT_smoke_inv`: coefficient `-0.000043` (lowers CT win probability)
- `lag_01__active_smokes_total`: coefficient `-0.000043` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000040` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000039` (lowers CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000039` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_OBSERVATION`: coefficient `0.003099` (raises CT win probability)
- `lag_07__T_place_OBSERVATION`: coefficient `-0.002747` (lowers CT win probability)
- `lag_05__CT_place_DECON`: coefficient `0.002683` (raises CT win probability)
- `lag_04__T_place_RAMP`: coefficient `-0.002443` (lowers CT win probability)
- `lag_07__CT_place_VENTS`: coefficient `-0.002258` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002188` (raises CT win probability)
- `lag_03__CT_place_SECRET`: coefficient `-0.002127` (lowers CT win probability)
- `lag_03__T_place_RAMP`: coefficient `-0.001980` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001775` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `0.001742` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113999`, seconds `28.00`, LSTM delta `+0.2616`

Top all feature movements:
- `lag_05__CT_place_DECON`: contribution `+0.042661`
- `lag_03__CT_place_SECRET`: contribution `+0.021892`
- `lag_04__T_place_RAMP`: contribution `+0.017282`
- `lag_11__CT_place_SECRET`: contribution `+0.016052`
- `lag_14__CT_place_VENTS`: contribution `+0.012411`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114671`, seconds `38.50`, LSTM delta `+0.2300`

Top all feature movements:
- `lag_07__T_place_OBSERVATION`: contribution `+0.046523`
- `lag_14__CT_place_DECON`: contribution `+0.027089`
- `lag_12__T_place_OBSERVATION`: contribution `+0.021418`
- `lag_07__CT_place_VENTS`: contribution `+0.018950`
- `lag_00__T_place_OBSERVATION`: contribution `+0.017406`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115119`, seconds `45.50`, LSTM delta `-0.1449`

Top all feature movements:
- `lag_14__T_place_OBSERVATION`: contribution `-0.052479`
- `lag_00__kill_diff_last_3s`: contribution `-0.005267`
- `lag_08__CT_place_TUNNELS`: contribution `-0.004136`
- `lag_15__T4__is_walking`: contribution `-0.003699`
- `lag_08__CT_kills_last_3s`: contribution `-0.003413`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113935`, seconds `27.00`, LSTM delta `+0.1316`

Top all feature movements:
- `lag_07__CT_place_VENTS`: contribution `+0.018950`
- `lag_03__CT_place_DECON`: contribution `+0.018463`
- `lag_01__CT_place_SECRET`: contribution `+0.011618`
- `lag_09__CT_place_SECRET`: contribution `+0.007161`
- `lag_02__T_place_RAMP`: contribution `+0.007071`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114127`, seconds `30.00`, LSTM delta `-0.1190`

Top all feature movements:
- `lag_09__CT_place_DECON`: contribution `-0.026460`
- `lag_15__CT_place_SECRET`: contribution `-0.014592`
- `lag_07__CT_place_SECRET`: contribution `-0.009104`
- `lag_00__kill_diff_last_3s`: contribution `-0.005267`
- `lag_06__CT_place_CATWALK`: contribution `-0.004864`

Top utility-only movements:
- No utility movement among the top local contributors.
