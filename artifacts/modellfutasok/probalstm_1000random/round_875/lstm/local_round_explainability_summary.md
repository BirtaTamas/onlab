# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `63282`, seconds `84.00`, LSTM `0.0779`, delta `-0.2418`
- tick `62962`, seconds `79.00`, LSTM `0.2633`, delta `-0.2317`
- tick `63250`, seconds `83.50`, LSTM `0.3197`, delta `+0.0837`
- tick `63090`, seconds `81.00`, LSTM `0.2220`, delta `-0.0614`
- tick `62610`, seconds `73.50`, LSTM `0.5217`, delta `+0.0518`
- tick `61202`, seconds `51.50`, LSTM `0.4741`, delta `+0.0340`
- tick `58930`, seconds `16.00`, LSTM `0.4496`, delta `-0.0297`
- tick `59730`, seconds `28.50`, LSTM `0.4660`, delta `-0.0297`
- tick `63058`, seconds `80.50`, LSTM `0.2834`, delta `+0.0273`
- tick `60562`, seconds `41.50`, LSTM `0.4651`, delta `+0.0269`

## Top 15 local ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.002114`, |coef| `0.002114`
- `lag_06__CT_shots_fired_sum`: coefficient `0.001582`, |coef| `0.001582`
- `lag_02__T5__shots_fired`: coefficient `-0.001562`, |coef| `0.001562`
- `lag_09__T_shots_fired_sum`: coefficient `0.001544`, |coef| `0.001544`
- `lag_05__T5__duck_amount`: coefficient `0.001517`, |coef| `0.001517`
- `lag_04__T1__shots_fired`: coefficient `0.001455`, |coef| `0.001455`
- `lag_00__CT2__is_walking`: coefficient `-0.001414`, |coef| `0.001414`
- `lag_13__CT_place_LIBRARY`: coefficient `0.001334`, |coef| `0.001334`
- `lag_06__CT5__shots_fired`: coefficient `0.001304`, |coef| `0.001304`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001299`, |coef| `0.001299`
- `lag_01__T5__shots_fired`: coefficient `-0.001292`, |coef| `0.001292`
- `lag_00__T3__is_walking`: coefficient `-0.001196`, |coef| `0.001196`
- `lag_11__CT4__is_scoped`: coefficient `0.001187`, |coef| `0.001187`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001182`, |coef| `0.001182`
- `lag_04__T5__shots_fired`: coefficient `-0.001163`, |coef| `0.001163`

## Top 10 utility ridge features

- `lag_08__T_B_site_active_infernos`: coefficient `0.001049` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000897` (raises CT win probability)
- `lag_08__T_active_infernos`: coefficient `0.000872` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.000864` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.000834` (raises CT win probability)
- `lag_03__CT5__molly`: coefficient `0.000788` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.000761` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.000735` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000728` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000725` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.002114` (lowers CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `0.001582` (raises CT win probability)
- `lag_02__T5__shots_fired`: coefficient `-0.001562` (lowers CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.001544` (raises CT win probability)
- `lag_05__T5__duck_amount`: coefficient `0.001517` (raises CT win probability)
- `lag_04__T1__shots_fired`: coefficient `0.001455` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001414` (lowers CT win probability)
- `lag_13__CT_place_LIBRARY`: coefficient `0.001334` (raises CT win probability)
- `lag_06__CT5__shots_fired`: coefficient `0.001304` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001299` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63282`, seconds `84.00`, LSTM delta `-0.2418`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `-0.033572`
- `lag_06__CT_shots_fired_sum`: contribution `-0.024183`
- `lag_06__CT5__shots_fired`: contribution `-0.015167`
- `lag_02__T_shots_fired_sum`: contribution `-0.012678`
- `lag_09__T4__shots_fired`: contribution `-0.008974`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62962`, seconds `79.00`, LSTM delta `-0.2317`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.014263`
- `lag_01__T_shots_fired_sum`: contribution `-0.009742`
- `lag_00__T_shots_fired_sum`: contribution `-0.008865`
- `lag_13__CT_place_LIBRARY`: contribution `-0.008555`
- `lag_04__T1__shots_fired`: contribution `-0.007825`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.002967`

### tick `63250`, seconds `83.50`, LSTM delta `+0.0837`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.023187`
- `lag_09__T_shots_fired_sum`: contribution `+0.011577`
- `lag_05__CT_shots_fired_sum`: contribution `+0.010737`
- `lag_08__T5__shots_fired`: contribution `+0.008363`
- `lag_01__T_shots_fired_sum`: contribution `-0.007794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63090`, seconds `81.00`, LSTM delta `-0.0614`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.013280`
- `lag_08__T_shots_fired_sum`: contribution `+0.007196`
- `lag_05__T_shots_fired_sum`: contribution `-0.006892`
- `lag_04__CT_shots_fired_sum`: contribution `-0.006349`
- `lag_06__T_shots_fired_sum`: contribution `-0.006201`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.002967`
- `lag_12__T_B_site_active_infernos`: contribution `-0.002536`

### tick `62610`, seconds `73.50`, LSTM delta `+0.0518`

Top all feature movements:
- `lag_07__CT_place_LIBRARY`: contribution `+0.005991`
- `lag_12__T_B_site_active_infernos`: contribution `+0.005072`
- `lag_02__CT_place_LIBRARY`: contribution `+0.004323`
- `lag_14__CT_place_BALCONY`: contribution `+0.003199`
- `lag_12__T_active_infernos`: contribution `+0.002908`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `+0.005072`
- `lag_12__T_active_infernos`: contribution `+0.002908`
- `lag_07__T_B_site_active_infernos`: contribution `+0.001073`
