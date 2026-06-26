# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `58068`, seconds `31.50`, LSTM `0.0558`, delta `-0.2850`
- tick `57940`, seconds `29.50`, LSTM `0.2872`, delta `+0.2237`
- tick `57044`, seconds `15.50`, LSTM `0.1438`, delta `-0.1939`
- tick `57684`, seconds `25.50`, LSTM `0.0327`, delta `-0.0613`
- tick `58004`, seconds `30.50`, LSTM `0.3039`, delta `+0.0482`
- tick `57076`, seconds `16.00`, LSTM `0.1001`, delta `-0.0436`
- tick `58036`, seconds `31.00`, LSTM `0.3408`, delta `+0.0368`
- tick `57972`, seconds `30.00`, LSTM `0.2557`, delta `-0.0315`
- tick `57588`, seconds `24.00`, LSTM `0.1013`, delta `-0.0258`
- tick `57876`, seconds `28.50`, LSTM `0.0409`, delta `+0.0256`

## Top 15 local ridge features

- `lag_04__T3__flash_duration`: coefficient `0.002033`, |coef| `0.002033`
- `lag_04__T_flash_duration_sum`: coefficient `0.001748`, |coef| `0.001748`
- `lag_08__T_flash_duration_sum`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_00__kill_diff_last_3s`: coefficient `0.001554`, |coef| `0.001554`
- `lag_08__T2__flash_duration`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_08__T3__flash_duration`: coefficient `-0.001365`, |coef| `0.001365`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001329`, |coef| `0.001329`
- `lag_08__T_flashed_players`: coefficient `-0.001325`, |coef| `0.001325`
- `lag_02__T_flashes_last_5s`: coefficient `0.001314`, |coef| `0.001314`
- `lag_00__T_place_LONGA`: coefficient `-0.001260`, |coef| `0.001260`
- `lag_12__CT_place_HOLE`: coefficient `0.001248`, |coef| `0.001248`
- `lag_04__T_flashed_players`: coefficient `0.001161`, |coef| `0.001161`
- `lag_06__T_flashes_last_5s`: coefficient `-0.001152`, |coef| `0.001152`
- `lag_02__T2__flash_duration`: coefficient `-0.001098`, |coef| `0.001098`
- `lag_02__T_place_LONGA`: coefficient `-0.001096`, |coef| `0.001096`

## Top 10 utility ridge features

- `lag_04__T3__flash_duration`: coefficient `0.002033` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.001748` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.001564` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.001378` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.001365` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.001314` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.001152` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.001098` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.001044` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `-0.000953` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001554` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.001329` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `-0.001325` (lowers CT win probability)
- `lag_00__T_place_LONGA`: coefficient `-0.001260` (lowers CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `0.001248` (raises CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.001161` (raises CT win probability)
- `lag_02__T_place_LONGA`: coefficient `-0.001096` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001089` (lowers CT win probability)
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `-0.001071` (lowers CT win probability)
- `lag_15__CT_place_UPPERTUNNEL`: coefficient `-0.001066` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58068`, seconds `31.50`, LSTM delta `-0.2850`

Top all feature movements:
- `lag_08__T_flash_duration_sum`: contribution `-0.015916`
- `lag_04__T3__flash_duration`: contribution `-0.014831`
- `lag_03__CT_shots_fired_sum`: contribution `-0.012006`
- `lag_06__T_flashes_last_5s`: contribution `-0.010442`
- `lag_08__T2__flash_duration`: contribution `-0.010230`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `-0.015916`
- `lag_04__T3__flash_duration`: contribution `-0.014831`
- `lag_06__T_flashes_last_5s`: contribution `-0.010442`
- `lag_08__T2__flash_duration`: contribution `-0.010230`
- `lag_08__T3__flash_duration`: contribution `-0.009726`

### tick `57940`, seconds `29.50`, LSTM delta `+0.2237`

Top all feature movements:
- `lag_04__T_flash_duration_sum`: contribution `+0.017786`
- `lag_04__T3__flash_duration`: contribution `+0.014484`
- `lag_02__T_flashes_last_5s`: contribution `+0.011907`
- `lag_04__T_flashed_players`: contribution `+0.008963`
- `lag_02__T2__flash_duration`: contribution `+0.008150`

Top utility-only movements:
- `lag_04__T_flash_duration_sum`: contribution `+0.017786`
- `lag_04__T3__flash_duration`: contribution `+0.014484`
- `lag_02__T_flashes_last_5s`: contribution `+0.011907`
- `lag_02__T2__flash_duration`: contribution `+0.008150`
- `lag_00__T3__flash_duration`: contribution `+0.006909`

### tick `57044`, seconds `15.50`, LSTM delta `-0.1939`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `-0.013928`
- `lag_14__CT_place_HOLE`: contribution `-0.010218`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `-0.006339`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `-0.005973`
- `lag_08__T_flashed_players`: contribution `-0.005112`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.003885`
- `lag_08__T_flash_duration_sum`: contribution `-0.002999`

### tick `57684`, seconds `25.50`, LSTM delta `-0.0613`

Top all feature movements:
- `lag_03__CT_place_UPPERTUNNEL`: contribution `-0.004104`
- `lag_00__kill_diff_last_3s`: contribution `-0.003741`
- `lag_00__T_kills_last_3s`: contribution `-0.003451`
- `lag_13__CT_place_EXTENDEDA`: contribution `-0.002715`
- `lag_10__T_place_LONGA`: contribution `-0.002142`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.000770`
- `lag_00__CT4__molly`: contribution `-0.000769`

### tick `58004`, seconds `30.50`, LSTM delta `+0.0482`

Top all feature movements:
- `lag_06__T_flash_duration_sum`: contribution `+0.007426`
- `lag_06__T2__flash_duration`: contribution `+0.006550`
- `lag_04__T_flash_duration_sum`: contribution `-0.005311`
- `lag_06__T_flashed_players`: contribution `+0.005077`
- `lag_02__T_place_LONGA`: contribution `+0.004671`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `+0.007426`
- `lag_06__T2__flash_duration`: contribution `+0.006550`
- `lag_04__T_flash_duration_sum`: contribution `-0.005311`
- `lag_04__T2__flash_duration`: contribution `-0.004380`
- `lag_04__T_flashes_last_5s`: contribution `+0.003506`
