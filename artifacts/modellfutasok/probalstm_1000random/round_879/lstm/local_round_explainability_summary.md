# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `68090`, seconds `96.00`, LSTM `0.1991`, delta `-0.2507`
- tick `63418`, seconds `23.00`, LSTM `0.5030`, delta `+0.2063`
- tick `63290`, seconds `21.00`, LSTM `0.4166`, delta `-0.1694`
- tick `67194`, seconds `82.00`, LSTM `0.5871`, delta `+0.1251`
- tick `67578`, seconds `88.00`, LSTM `0.5293`, delta `-0.1001`
- tick `64058`, seconds `33.00`, LSTM `0.4381`, delta `+0.0786`
- tick `68122`, seconds `96.50`, LSTM `0.1223`, delta `-0.0768`
- tick `63834`, seconds `29.50`, LSTM `0.4206`, delta `-0.0588`
- tick `64506`, seconds `40.00`, LSTM `0.4150`, delta `-0.0559`
- tick `67386`, seconds `85.00`, LSTM `0.6621`, delta `+0.0556`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002200`, |coef| `0.002200`
- `lag_00__damage_diff_last_5s`: coefficient `0.002077`, |coef| `0.002077`
- `lag_10__T1__duck_amount`: coefficient `0.001926`, |coef| `0.001926`
- `lag_02__T_place_RAMP`: coefficient `-0.001754`, |coef| `0.001754`
- `lag_04__CT_place_HUT`: coefficient `0.001745`, |coef| `0.001745`
- `lag_15__CT4__flash_duration`: coefficient `-0.001663`, |coef| `0.001663`
- `lag_00__T_kills_last_3s`: coefficient `-0.001641`, |coef| `0.001641`
- `lag_11__CT_place_SECRET`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_01__T_place_RAMP`: coefficient `-0.001573`, |coef| `0.001573`
- `lag_00__CT_place_DECON`: coefficient `0.001551`, |coef| `0.001551`
- `lag_11__CT_place_HUT`: coefficient `-0.001515`, |coef| `0.001515`
- `lag_04__CT_place_LOBBY`: coefficient `-0.001317`, |coef| `0.001317`
- `lag_11__CT2__duck_amount`: coefficient `-0.001308`, |coef| `0.001308`
- `lag_01__CT_place_DECON`: coefficient `0.001273`, |coef| `0.001273`
- `lag_02__T2__flash_duration`: coefficient `-0.001270`, |coef| `0.001270`

## Top 10 utility ridge features

- `lag_15__CT4__flash_duration`: coefficient `-0.001663` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.001270` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `0.001092` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000978` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000949` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000943` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000925` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000916` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000895` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000875` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002200` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002077` (raises CT win probability)
- `lag_10__T1__duck_amount`: coefficient `0.001926` (raises CT win probability)
- `lag_02__T_place_RAMP`: coefficient `-0.001754` (lowers CT win probability)
- `lag_04__CT_place_HUT`: coefficient `0.001745` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001641` (lowers CT win probability)
- `lag_11__CT_place_SECRET`: coefficient `-0.001594` (lowers CT win probability)
- `lag_01__T_place_RAMP`: coefficient `-0.001573` (lowers CT win probability)
- `lag_00__CT_place_DECON`: coefficient `0.001551` (raises CT win probability)
- `lag_11__CT_place_HUT`: coefficient `-0.001515` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `68090`, seconds `96.00`, LSTM delta `-0.2507`

Top all feature movements:
- `lag_02__T_place_DECON`: contribution `-0.019951`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.017029`
- `lag_04__CT_place_HUT`: contribution `-0.017021`
- `lag_11__CT_place_HUT`: contribution `-0.014777`
- `lag_04__CT_place_LOBBY`: contribution `-0.010783`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `-0.008672`
- `lag_02__T_flash_duration_sum`: contribution `-0.002623`

### tick `63418`, seconds `23.00`, LSTM delta `+0.2063`

Top all feature movements:
- `lag_15__CT4__flash_duration`: contribution `+0.011329`
- `lag_06__T_place_ROOF`: contribution `+0.006810`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006031`
- `lag_03__T_shots_fired_sum`: contribution `+0.005966`
- `lag_00__kill_diff_last_3s`: contribution `+0.005296`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `+0.011329`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.003451`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.003261`
- `lag_10__T_A_site_active_infernos`: contribution `+0.002753`
- `lag_10__T_B_site_active_infernos`: contribution `+0.002473`

### tick `63290`, seconds `21.00`, LSTM delta `-0.1694`

Top all feature movements:
- `lag_10__T1__duck_amount`: contribution `-0.007539`
- `lag_11__CT4__flash_duration`: contribution `-0.007435`
- `lag_15__CT_place_HUTROOF`: contribution `-0.005905`
- `lag_00__kill_diff_last_3s`: contribution `-0.005296`
- `lag_00__T_kills_last_3s`: contribution `-0.005200`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `-0.007435`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.003233`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.003074`
- `lag_00__CT5__utility_total`: contribution `-0.002671`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.002461`

### tick `67194`, seconds `82.00`, LSTM delta `+0.1251`

Top all feature movements:
- `lag_11__CT_place_SECRET`: contribution `+0.016405`
- `lag_02__T_place_RAMP`: contribution `+0.012404`
- `lag_00__kill_diff_last_3s`: contribution `+0.005296`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005277`
- `lag_11__CT2__duck_amount`: contribution `+0.004438`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67578`, seconds `88.00`, LSTM delta `-0.1001`

Top all feature movements:
- `lag_00__CT_place_DECON`: contribution `-0.024657`
- `lag_06__CT_place_OBSERVATION`: contribution `-0.010481`
- `lag_01__T_place_RAMP`: contribution `+0.005563`
- `lag_00__kill_diff_last_3s`: contribution `-0.005296`
- `lag_00__T_kills_last_3s`: contribution `-0.005200`

Top utility-only movements:
- No utility movement among the top local contributors.
