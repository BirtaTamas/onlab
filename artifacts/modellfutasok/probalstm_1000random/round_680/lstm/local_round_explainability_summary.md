# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `168746`, seconds `58.00`, LSTM `0.7087`, delta `-0.1826`
- tick `169642`, seconds `72.00`, LSTM `0.7724`, delta `+0.1759`
- tick `168490`, seconds `54.00`, LSTM `0.8690`, delta `+0.1320`
- tick `169706`, seconds `73.00`, LSTM `0.9309`, delta `+0.0910`
- tick `169578`, seconds `71.00`, LSTM `0.6131`, delta `-0.0697`
- tick `169674`, seconds `72.50`, LSTM `0.8399`, delta `+0.0675`
- tick `165610`, seconds `9.00`, LSTM `0.6396`, delta `-0.0500`
- tick `168874`, seconds `60.00`, LSTM `0.7261`, delta `+0.0461`
- tick `168714`, seconds `57.50`, LSTM `0.8913`, delta `+0.0461`
- tick `166762`, seconds `27.00`, LSTM `0.7479`, delta `+0.0391`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002492`, |coef| `0.002492`
- `lag_13__CT_flashes_last_5s`: coefficient `0.002337`, |coef| `0.002337`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.002157`, |coef| `0.002157`
- `lag_03__CT_flashes_last_5s`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_00__kill_diff_last_3s`: coefficient `0.001936`, |coef| `0.001936`
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `-0.001699`, |coef| `0.001699`
- `lag_00__CT_kills_last_3s`: coefficient `0.001581`, |coef| `0.001581`
- `lag_12__CT_place_LONGDOORS`: coefficient `0.001315`, |coef| `0.001315`
- `lag_15__T_place_TUNNELSTAIRS`: coefficient `-0.001312`, |coef| `0.001312`
- `lag_00__damage_diff_last_5s`: coefficient `0.001247`, |coef| `0.001247`
- `lag_00__T_burning_players`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_12__CT_place_PIT`: coefficient `-0.001197`, |coef| `0.001197`
- `lag_02__CT5__duck_amount`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_09__CT_place_ARAMP`: coefficient `-0.001172`, |coef| `0.001172`
- `lag_08__T_place_MIDDOORS`: coefficient `0.001163`, |coef| `0.001163`

## Top 10 utility ridge features

- `lag_13__CT_flashes_last_5s`: coefficient `0.002337` (raises CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `-0.002020` (lowers CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000955` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000938` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000916` (lowers CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.000913` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.000834` (lowers CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `0.000815` (raises CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `-0.000739` (lowers CT win probability)
- `lag_08__T4__smoke`: coefficient `-0.000690` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002492` (raises CT win probability)
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.002157` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001936` (raises CT win probability)
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `-0.001699` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001581` (raises CT win probability)
- `lag_12__CT_place_LONGDOORS`: coefficient `0.001315` (raises CT win probability)
- `lag_15__T_place_TUNNELSTAIRS`: coefficient `-0.001312` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001247` (raises CT win probability)
- `lag_00__T_burning_players`: coefficient `-0.001213` (lowers CT win probability)
- `lag_12__CT_place_PIT`: coefficient `-0.001197` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `168746`, seconds `58.00`, LSTM delta `-0.1826`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012119`
- `lag_03__CT_place_HOLE`: contribution `-0.009411`
- `lag_00__kill_diff_last_3s`: contribution `-0.004660`
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.004270`
- `lag_01__T1__duck_amount`: contribution `-0.003998`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `169642`, seconds `72.00`, LSTM delta `+0.1759`

Top all feature movements:
- `lag_13__CT_flashes_last_5s`: contribution `+0.025700`
- `lag_03__CT_flashes_last_5s`: contribution `+0.022215`
- `lag_09__CT_place_OUTSIDELONG`: contribution `+0.021873`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.011859`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008657`

Top utility-only movements:
- `lag_13__CT_flashes_last_5s`: contribution `+0.025700`
- `lag_03__CT_flashes_last_5s`: contribution `+0.022215`
- `lag_10__T1__flash_duration`: contribution `+0.006445`
- `lag_10__T4__flash_duration`: contribution `+0.005727`
- `lag_10__T_flash_duration_sum`: contribution `+0.003721`

### tick `168490`, seconds `54.00`, LSTM delta `+0.1320`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008657`
- `lag_09__CT_place_ARAMP`: contribution `+0.007300`
- `lag_12__CT_place_LONGDOORS`: contribution `+0.005760`
- `lag_12__CT_place_PIT`: contribution `+0.005153`
- `lag_08__T_place_MIDDOORS`: contribution `+0.004945`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `169706`, seconds `73.00`, LSTM delta `+0.0910`

Top all feature movements:
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.009163`
- `lag_15__CT_flashes_last_5s`: contribution `+0.008966`
- `lag_11__CT_place_OUTSIDELONG`: contribution `+0.007605`
- `lag_05__CT_flashes_last_5s`: contribution `+0.007253`
- `lag_01__CT_place_OUTSIDELONG`: contribution `+0.006234`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.008966`
- `lag_05__CT_flashes_last_5s`: contribution `+0.007253`
- `lag_12__T1__flash_duration`: contribution `+0.003649`
- `lag_12__T4__flash_duration`: contribution `+0.003020`
- `lag_12__T_flash_duration_sum`: contribution `+0.002373`

### tick `169578`, seconds `71.00`, LSTM delta `-0.0697`

Top all feature movements:
- `lag_01__CT_flashes_last_5s`: contribution `-0.010497`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.009163`
- `lag_11__CT_flashes_last_5s`: contribution `-0.008131`
- `lag_12__T_place_TUNNELSTAIRS`: contribution `-0.007957`
- `lag_07__CT_place_OUTSIDELONG`: contribution `-0.006904`

Top utility-only movements:
- `lag_01__CT_flashes_last_5s`: contribution `-0.010497`
- `lag_11__CT_flashes_last_5s`: contribution `-0.008131`
- `lag_08__T4__flash_duration`: contribution `-0.002315`
- `lag_08__T1__flash_duration`: contribution `-0.001944`
