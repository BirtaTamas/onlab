# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `18`

## Largest probability jumps

- tick `139358`, seconds `132.00`, LSTM `0.1648`, delta `-0.2902`
- tick `139038`, seconds `127.00`, LSTM `0.4737`, delta `+0.2520`
- tick `138974`, seconds `126.00`, LSTM `0.2615`, delta `-0.1869`
- tick `139070`, seconds `127.50`, LSTM `0.3934`, delta `-0.0803`
- tick `137278`, seconds `99.50`, LSTM `0.5188`, delta `-0.0754`
- tick `138910`, seconds `125.00`, LSTM `0.4356`, delta `-0.0600`
- tick `135966`, seconds `79.00`, LSTM `0.4797`, delta `-0.0525`
- tick `137118`, seconds `97.00`, LSTM `0.5662`, delta `+0.0490`
- tick `139198`, seconds `129.50`, LSTM `0.4610`, delta `+0.0460`
- tick `135198`, seconds `67.00`, LSTM `0.5478`, delta `-0.0414`

## Top 15 local ridge features

- `lag_09__CT_place_VENTS`: coefficient `-0.003533`, |coef| `0.003533`
- `lag_00__kill_diff_last_3s`: coefficient `0.003278`, |coef| `0.003278`
- `lag_03__T2__shots_fired`: coefficient `-0.002925`, |coef| `0.002925`
- `lag_03__T_shots_fired_sum`: coefficient `-0.002812`, |coef| `0.002812`
- `lag_00__T_kills_last_3s`: coefficient `-0.002762`, |coef| `0.002762`
- `lag_01__CT_place_VENTS`: coefficient `0.002648`, |coef| `0.002648`
- `lag_02__CT_place_RAMP`: coefficient `0.002026`, |coef| `0.002026`
- `lag_01__T3__is_walking`: coefficient `0.001796`, |coef| `0.001796`
- `lag_03__CT_place_RAMP`: coefficient `0.001722`, |coef| `0.001722`
- `lag_11__T3__is_walking`: coefficient `-0.001599`, |coef| `0.001599`
- `lag_10__CT_place_VENTS`: coefficient `-0.001593`, |coef| `0.001593`
- `lag_01__CT_place_RAMP`: coefficient `0.001546`, |coef| `0.001546`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001517`, |coef| `0.001517`
- `lag_03__CT_duck_amount_mean`: coefficient `-0.001507`, |coef| `0.001507`
- `lag_15__CT4__duck_amount`: coefficient `-0.001469`, |coef| `0.001469`

## Top 10 utility ridge features

- `lag_00__CT5__flash`: coefficient `0.000993` (raises CT win probability)
- `lag_12__CT5__flash`: coefficient `0.000727` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000701` (raises CT win probability)
- `lag_02__CT5__flash`: coefficient `-0.000624` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000516` (raises CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000472` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000444` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `0.000432` (raises CT win probability)
- `lag_13__T3__flash_duration`: coefficient `-0.000383` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000377` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_VENTS`: coefficient `-0.003533` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003278` (raises CT win probability)
- `lag_03__T2__shots_fired`: coefficient `-0.002925` (lowers CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.002812` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002762` (lowers CT win probability)
- `lag_01__CT_place_VENTS`: coefficient `0.002648` (raises CT win probability)
- `lag_02__CT_place_RAMP`: coefficient `0.002026` (raises CT win probability)
- `lag_01__T3__is_walking`: coefficient `0.001796` (raises CT win probability)
- `lag_03__CT_place_RAMP`: coefficient `0.001722` (raises CT win probability)
- `lag_11__T3__is_walking`: coefficient `-0.001599` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `139358`, seconds `132.00`, LSTM delta `-0.2902`

Top all feature movements:
- `lag_09__CT_place_VENTS`: contribution `-0.029647`
- `lag_01__CT_place_VENTS`: contribution `-0.022222`
- `lag_00__T_kills_last_3s`: contribution `-0.008749`
- `lag_00__kill_diff_last_3s`: contribution `-0.007890`
- `lag_13__T2__shots_fired`: contribution `-0.006819`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139038`, seconds `127.00`, LSTM delta `+0.2520`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.016869`
- `lag_03__T2__shots_fired`: contribution `+0.013770`
- `lag_00__kill_diff_last_3s`: contribution `+0.007890`
- `lag_06__T2__duck_amount`: contribution `+0.004636`
- `lag_05__CT4__duck_amount`: contribution `+0.004599`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `138974`, seconds `126.00`, LSTM delta `-0.1869`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008749`
- `lag_00__kill_diff_last_3s`: contribution `-0.007890`
- `lag_03__T_shots_fired_sum`: contribution `-0.006326`
- `lag_02__CT_place_RAMP`: contribution `-0.006053`
- `lag_03__T2__shots_fired`: contribution `-0.005164`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139070`, seconds `127.50`, LSTM delta `-0.0803`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.008434`
- `lag_03__T2__shots_fired`: contribution `-0.006885`
- `lag_04__T_shots_fired_sum`: contribution `-0.006679`
- `lag_04__T2__shots_fired`: contribution `-0.005510`
- `lag_04__CT5__duck_amount`: contribution `-0.003635`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137278`, seconds `99.50`, LSTM delta `-0.0754`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008749`
- `lag_00__kill_diff_last_3s`: contribution `-0.007890`
- `lag_11__CT5__is_walking`: contribution `-0.002915`
- `lag_09__CT_place_RAFTERS`: contribution `-0.002843`
- `lag_00__damage_diff_last_5s`: contribution `-0.002784`

Top utility-only movements:
- No utility movement among the top local contributors.
