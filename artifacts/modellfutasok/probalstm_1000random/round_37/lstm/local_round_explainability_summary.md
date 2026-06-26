# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `19`

## Largest probability jumps

- tick `170607`, seconds `96.50`, LSTM `0.7131`, delta `+0.1366`
- tick `171119`, seconds `104.50`, LSTM `0.9485`, delta `+0.1056`
- tick `170671`, seconds `97.50`, LSTM `0.8573`, delta `+0.1021`
- tick `170575`, seconds `96.00`, LSTM `0.5766`, delta `+0.0497`
- tick `166703`, seconds `35.50`, LSTM `0.5263`, delta `-0.0436`
- tick `170639`, seconds `97.00`, LSTM `0.7553`, delta `+0.0422`
- tick `169263`, seconds `75.50`, LSTM `0.4740`, delta `-0.0349`
- tick `167087`, seconds `41.50`, LSTM `0.5020`, delta `+0.0291`
- tick `166959`, seconds `39.50`, LSTM `0.4753`, delta `-0.0289`
- tick `170895`, seconds `101.00`, LSTM `0.8427`, delta `-0.0286`

## Top 15 local ridge features

- `lag_13__CT_place_LIBRARY`: coefficient `0.001451`, |coef| `0.001451`
- `lag_08__CT_place_BALCONY`: coefficient `-0.001169`, |coef| `0.001169`
- `lag_06__T4__flash_duration`: coefficient `0.001145`, |coef| `0.001145`
- `lag_14__CT_place_LIBRARY`: coefficient `0.001095`, |coef| `0.001095`
- `lag_00__CT_kills_last_3s`: coefficient `0.001014`, |coef| `0.001014`
- `lag_01__CT_damage_last_5s`: coefficient `0.000974`, |coef| `0.000974`
- `lag_01__CT_place_BALCONY`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000839`, |coef| `0.000839`
- `lag_07__T_place_BALCONY`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_11__CT_place_RUINS`: coefficient `0.000772`, |coef| `0.000772`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000769`, |coef| `0.000769`
- `lag_00__T3__flash_duration`: coefficient `-0.000748`, |coef| `0.000748`
- `lag_13__CT_place_PIT`: coefficient `-0.000741`, |coef| `0.000741`
- `lag_10__T_flashes_last_5s`: coefficient `0.000735`, |coef| `0.000735`
- `lag_00__T5__duck_amount`: coefficient `-0.000729`, |coef| `0.000729`

## Top 10 utility ridge features

- `lag_06__T4__flash_duration`: coefficient `0.001145` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000748` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000735` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.000698` (raises CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000664` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000653` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.000632` (raises CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `0.000628` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `0.000626` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000612` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_LIBRARY`: coefficient `0.001451` (raises CT win probability)
- `lag_08__CT_place_BALCONY`: coefficient `-0.001169` (lowers CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `0.001095` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001014` (raises CT win probability)
- `lag_01__CT_damage_last_5s`: coefficient `0.000974` (raises CT win probability)
- `lag_01__CT_place_BALCONY`: coefficient `-0.000923` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000839` (lowers CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.000776` (lowers CT win probability)
- `lag_11__CT_place_RUINS`: coefficient `0.000772` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000769` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `170607`, seconds `96.50`, LSTM delta `+0.1366`

Top all feature movements:
- `lag_13__CT_place_LIBRARY`: contribution `+0.009306`
- `lag_08__CT_place_BALCONY`: contribution `+0.007504`
- `lag_06__T4__flash_duration`: contribution `+0.007492`
- `lag_01__CT_place_BALCONY`: contribution `+0.005921`
- `lag_06__T3__flash_duration`: contribution `+0.004500`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.007492`
- `lag_06__T3__flash_duration`: contribution `+0.004500`
- `lag_06__T_flash_duration_sum`: contribution `+0.003270`
- `lag_09__T_B_site_active_smokes`: contribution `+0.001903`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.001802`

### tick `171119`, seconds `104.50`, LSTM delta `+0.1056`

Top all feature movements:
- `lag_14__CT_place_LIBRARY`: contribution `+0.007022`
- `lag_14__T3__flash_duration`: contribution `+0.004210`
- `lag_10__T4__flash_duration`: contribution `+0.003977`
- `lag_04__T_bomb_zone_count`: contribution `+0.003117`
- `lag_00__CT_kills_last_3s`: contribution `+0.002927`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `+0.004210`
- `lag_10__T4__flash_duration`: contribution `+0.003977`

### tick `170671`, seconds `97.50`, LSTM delta `+0.1021`

Top all feature movements:
- `lag_00__T3__flash_duration`: contribution `+0.004822`
- `lag_08__T3__flash_duration`: contribution `+0.004073`
- `lag_08__T4__flash_duration`: contribution `+0.003517`
- `lag_15__CT_place_LIBRARY`: contribution `+0.003493`
- `lag_10__CT_place_BALCONY`: contribution `+0.003302`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.004822`
- `lag_08__T3__flash_duration`: contribution `+0.004073`
- `lag_08__T4__flash_duration`: contribution `+0.003517`
- `lag_08__T_flash_duration_sum`: contribution `+0.002849`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.002010`

### tick `170575`, seconds `96.00`, LSTM delta `+0.0497`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `+0.005386`
- `lag_05__T4__flash_duration`: contribution `+0.003408`
- `lag_01__T_shots_fired_sum`: contribution `+0.002976`
- `lag_00__T5__duck_amount`: contribution `-0.002767`
- `lag_07__CT_place_BALCONY`: contribution `+0.002725`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.003408`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.002010`
- `lag_05__T3__flash_duration`: contribution `+0.001965`
- `lag_08__T_B_site_active_smokes`: contribution `+0.001269`

### tick `166703`, seconds `35.50`, LSTM delta `-0.0436`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `-0.009780`
- `lag_04__T_place_BALCONY`: contribution `-0.006392`
- `lag_00__CT_place_BALCONY`: contribution `-0.005386`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002670`
- `lag_01__T_shots_fired_sum`: contribution `+0.002480`

Top utility-only movements:
- No utility movement among the top local contributors.
