# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `59340`, seconds `35.50`, LSTM `0.2439`, delta `-0.2838`
- tick `62860`, seconds `90.50`, LSTM `0.8537`, delta `+0.2539`
- tick `59468`, seconds `37.50`, LSTM `0.7189`, delta `+0.2095`
- tick `59372`, seconds `36.00`, LSTM `0.4294`, delta `+0.1854`
- tick `61740`, seconds `73.00`, LSTM `0.4612`, delta `-0.1593`
- tick `61900`, seconds `75.50`, LSTM `0.6038`, delta `+0.1468`
- tick `61868`, seconds `75.00`, LSTM `0.4569`, delta `+0.1065`
- tick `58028`, seconds `15.00`, LSTM `0.3775`, delta `-0.0713`
- tick `61836`, seconds `74.50`, LSTM `0.3505`, delta `-0.0572`
- tick `59404`, seconds `36.50`, LSTM `0.4821`, delta `+0.0527`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004970`, |coef| `0.004970`
- `lag_00__CT_place_LIBRARY`: coefficient `0.003912`, |coef| `0.003912`
- `lag_00__CT_kills_last_3s`: coefficient `0.003670`, |coef| `0.003670`
- `lag_00__damage_diff_last_5s`: coefficient `0.003564`, |coef| `0.003564`
- `lag_01__damage_diff_last_5s`: coefficient `0.002938`, |coef| `0.002938`
- `lag_00__T_kills_last_3s`: coefficient `-0.002514`, |coef| `0.002514`
- `lag_00__T2__has_bomb`: coefficient `-0.002501`, |coef| `0.002501`
- `lag_00__CT_damage_last_5s`: coefficient `0.002100`, |coef| `0.002100`
- `lag_00__T4__is_walking`: coefficient `-0.002065`, |coef| `0.002065`
- `lag_01__CT_place_QUAD`: coefficient `0.002065`, |coef| `0.002065`
- `lag_00__T2__alive`: coefficient `-0.002062`, |coef| `0.002062`
- `lag_01__CT_place_LIBRARY`: coefficient `0.002057`, |coef| `0.002057`
- `lag_04__T_duck_amount_mean`: coefficient `-0.002025`, |coef| `0.002025`
- `lag_00__T_place_SECONDMID`: coefficient `0.001978`, |coef| `0.001978`
- `lag_01__kill_diff_last_3s`: coefficient `0.001942`, |coef| `0.001942`

## Top 10 utility ridge features

- `lag_00__T2__smoke`: coefficient `-0.001889` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.001782` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001463` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.001410` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001266` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.001226` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.001010` (lowers CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000989` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000972` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `0.000912` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004970` (raises CT win probability)
- `lag_00__CT_place_LIBRARY`: coefficient `0.003912` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003670` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003564` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002938` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002514` (lowers CT win probability)
- `lag_00__T2__has_bomb`: coefficient `-0.002501` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002100` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.002065` (lowers CT win probability)
- `lag_01__CT_place_QUAD`: coefficient `0.002065` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `59340`, seconds `35.50`, LSTM delta `-0.2838`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011962`
- `lag_04__CT2__flash_duration`: contribution `-0.011301`
- `lag_05__T1__flash_duration`: contribution `-0.010709`
- `lag_00__T_kills_last_3s`: contribution `-0.007965`
- `lag_00__T_shots_fired_sum`: contribution `-0.005960`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `-0.011301`
- `lag_05__T1__flash_duration`: contribution `-0.010709`
- `lag_05__T_flash_duration_sum`: contribution `-0.005725`
- `lag_05__T4__flash_duration`: contribution `-0.004699`

### tick `62860`, seconds `90.50`, LSTM delta `+0.2539`

Top all feature movements:
- `lag_00__CT_place_LIBRARY`: contribution `+0.025084`
- `lag_00__kill_diff_last_3s`: contribution `+0.011962`
- `lag_00__CT_kills_last_3s`: contribution `+0.010596`
- `lag_10__T_duck_amount_mean`: contribution `+0.008274`
- `lag_00__T2__has_bomb`: contribution `+0.007808`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `+0.004148`

### tick `59468`, seconds `37.50`, LSTM delta `+0.2095`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011962`
- `lag_00__CT_kills_last_3s`: contribution `+0.010596`
- `lag_03__T_shots_fired_sum`: contribution `+0.010170`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006882`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.006044`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.006044`
- `lag_09__T1__flash_duration`: contribution `+0.005798`
- `lag_00__T1__flash_duration`: contribution `+0.004234`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.004103`
- `lag_09__T_flash_duration_sum`: contribution `+0.003478`

### tick `59372`, seconds `36.00`, LSTM delta `+0.1854`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011962`
- `lag_00__T_shots_fired_sum`: contribution `+0.011921`
- `lag_00__CT_kills_last_3s`: contribution `+0.010596`
- `lag_06__T1__flash_duration`: contribution `+0.005115`
- `lag_01__kill_diff_last_3s`: contribution `-0.004674`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.005115`
- `lag_06__T_flash_duration_sum`: contribution `+0.003183`
- `lag_06__T4__flash_duration`: contribution `+0.002912`

### tick `61740`, seconds `73.00`, LSTM delta `-0.1593`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011962`
- `lag_00__T_kills_last_3s`: contribution `-0.007965`
- `lag_14__CT_place_PIT`: contribution `-0.007879`
- `lag_00__T_place_SECONDMID`: contribution `-0.006476`
- `lag_00__damage_diff_last_5s`: contribution `-0.005870`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.002399`
