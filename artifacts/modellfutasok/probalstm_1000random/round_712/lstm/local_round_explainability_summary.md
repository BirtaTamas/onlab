# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `28557`, seconds `33.50`, LSTM `0.3913`, delta `+0.2580`
- tick `27917`, seconds `23.50`, LSTM `0.2774`, delta `-0.2221`
- tick `32141`, seconds `89.50`, LSTM `0.8496`, delta `+0.1882`
- tick `28589`, seconds `34.00`, LSTM `0.4990`, delta `+0.1077`
- tick `29517`, seconds `48.50`, LSTM `0.6475`, delta `+0.0930`
- tick `28109`, seconds `26.50`, LSTM `0.3079`, delta `+0.0763`
- tick `28269`, seconds `29.00`, LSTM `0.2725`, delta `-0.0598`
- tick `28301`, seconds `29.50`, LSTM `0.2131`, delta `-0.0594`
- tick `32205`, seconds `90.50`, LSTM `0.9637`, delta `+0.0579`
- tick `30669`, seconds `66.50`, LSTM `0.6701`, delta `-0.0574`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003463`, |coef| `0.003463`
- `lag_00__kill_diff_last_3s`: coefficient `0.003403`, |coef| `0.003403`
- `lag_00__CT_kills_last_3s`: coefficient `0.002878`, |coef| `0.002878`
- `lag_00__damage_diff_last_5s`: coefficient `0.002706`, |coef| `0.002706`
- `lag_00__T5__shots_fired`: coefficient `0.002182`, |coef| `0.002182`
- `lag_02__T2__shots_fired`: coefficient `0.002181`, |coef| `0.002181`
- `lag_01__T2__shots_fired`: coefficient `0.002051`, |coef| `0.002051`
- `lag_00__CT_damage_last_5s`: coefficient `0.001988`, |coef| `0.001988`
- `lag_14__CT5__flash_duration`: coefficient `0.001940`, |coef| `0.001940`
- `lag_00__T2__duck_amount`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001834`, |coef| `0.001834`
- `lag_01__kill_diff_last_3s`: coefficient `0.001787`, |coef| `0.001787`
- `lag_02__T_place_BALCONY`: coefficient `0.001744`, |coef| `0.001744`
- `lag_15__CT_place_BALCONY`: coefficient `-0.001726`, |coef| `0.001726`
- `lag_00__CT2__shots_fired`: coefficient `0.001681`, |coef| `0.001681`

## Top 10 utility ridge features

- `lag_14__CT5__flash_duration`: coefficient `0.001940` (raises CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `0.001416` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001346` (lowers CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.001281` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001185` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.001162` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001089` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.001082` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.001029` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000994` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003463` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003403` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002878` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002706` (raises CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.002182` (raises CT win probability)
- `lag_02__T2__shots_fired`: coefficient `0.002181` (raises CT win probability)
- `lag_01__T2__shots_fired`: coefficient `0.002051` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001988` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001889` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001834` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28557`, seconds `33.50`, LSTM delta `+0.2580`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012030`
- `lag_00__CT_kills_last_3s`: contribution `+0.008308`
- `lag_00__kill_diff_last_3s`: contribution `+0.008191`
- `lag_00__T2__duck_amount`: contribution `+0.007221`
- `lag_00__damage_diff_last_5s`: contribution `+0.006105`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.005457`
- `lag_04__CT_B_site_active_infernos`: contribution `+0.004399`
- `lag_01__T_B_site_active_infernos`: contribution `+0.003058`

### tick `27917`, seconds `23.50`, LSTM delta `-0.2221`

Top all feature movements:
- `lag_14__CT5__flash_duration`: contribution `-0.013870`
- `lag_04__T_flashed_players`: contribution `-0.008494`
- `lag_00__kill_diff_last_3s`: contribution `-0.008191`
- `lag_02__T_flashed_players`: contribution `-0.007339`
- `lag_05__CT_place_ARCH`: contribution `-0.006188`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `-0.013870`
- `lag_14__CT_flash_duration_sum`: contribution `-0.002952`

### tick `32141`, seconds `89.50`, LSTM delta `+0.1882`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.023985`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012030`
- `lag_15__CT_place_BALCONY`: contribution `+0.011080`
- `lag_00__CT_kills_last_3s`: contribution `+0.008308`
- `lag_00__kill_diff_last_3s`: contribution `+0.008191`

Top utility-only movements:
- `lag_00__T3__molly`: contribution `+0.002581`

### tick `28589`, seconds `34.00`, LSTM delta `+0.1077`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012030`
- `lag_02__T2__shots_fired`: contribution `+0.006415`
- `lag_01__CT_shots_fired_sum`: contribution `+0.006369`
- `lag_01__T2__shots_fired`: contribution `+0.004826`
- `lag_07__T1__duck_amount`: contribution `-0.004796`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.004070`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.003197`

### tick `29517`, seconds `48.50`, LSTM delta `+0.0930`

Top all feature movements:
- `lag_15__CT_place_BALCONY`: contribution `+0.011080`
- `lag_00__CT_kills_last_3s`: contribution `+0.008308`
- `lag_00__kill_diff_last_3s`: contribution `+0.008191`
- `lag_00__damage_diff_last_5s`: contribution `+0.004518`
- `lag_00__CT_damage_last_5s`: contribution `+0.004333`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.002322`
