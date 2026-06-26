# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `44023`, seconds `22.00`, LSTM `0.3815`, delta `-0.2400`
- tick `44343`, seconds `27.00`, LSTM `0.0642`, delta `-0.1405`
- tick `44055`, seconds `22.50`, LSTM `0.2562`, delta `-0.1253`
- tick `43767`, seconds `18.00`, LSTM `0.6367`, delta `+0.0826`
- tick `43799`, seconds `18.50`, LSTM `0.5697`, delta `-0.0670`
- tick `44215`, seconds `25.00`, LSTM `0.2643`, delta `+0.0599`
- tick `43639`, seconds `16.00`, LSTM `0.5421`, delta `-0.0531`
- tick `43991`, seconds `21.50`, LSTM `0.6215`, delta `+0.0509`
- tick `44087`, seconds `23.00`, LSTM `0.2203`, delta `-0.0359`
- tick `44279`, seconds `26.00`, LSTM `0.2037`, delta `-0.0337`

## Top 15 local ridge features

- `lag_08__T_place_MINI`: coefficient `0.002025`, |coef| `0.002025`
- `lag_07__CT_place_HUT`: coefficient `0.001653`, |coef| `0.001653`
- `lag_12__CT_place_HUT`: coefficient `-0.001581`, |coef| `0.001581`
- `lag_13__CT_place_HUT`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_11__T5__flash_duration`: coefficient `0.001210`, |coef| `0.001210`
- `lag_10__T_place_HUT`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_08__CT_place_HUT`: coefficient `0.001156`, |coef| `0.001156`
- `lag_11__T_burning_players`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_08__T4__duck_amount`: coefficient `-0.001043`, |coef| `0.001043`
- `lag_14__CT4__flash_duration`: coefficient `0.000995`, |coef| `0.000995`
- `lag_12__T_place_ROOF`: coefficient `0.000987`, |coef| `0.000987`
- `lag_00__T_kills_last_3s`: coefficient `-0.000983`, |coef| `0.000983`
- `lag_12__T_burning_players`: coefficient `-0.000955`, |coef| `0.000955`
- `lag_10__CT5__duck_amount`: coefficient `0.000954`, |coef| `0.000954`
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000933`, |coef| `0.000933`

## Top 10 utility ridge features

- `lag_11__T5__flash_duration`: coefficient `0.001210` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.000995` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000933` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000856` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.000761` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000749` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000716` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `-0.000640` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000637` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000627` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_MINI`: coefficient `0.002025` (raises CT win probability)
- `lag_07__CT_place_HUT`: coefficient `0.001653` (raises CT win probability)
- `lag_12__CT_place_HUT`: coefficient `-0.001581` (lowers CT win probability)
- `lag_13__CT_place_HUT`: coefficient `-0.001510` (lowers CT win probability)
- `lag_10__T_place_HUT`: coefficient `-0.001184` (lowers CT win probability)
- `lag_08__CT_place_HUT`: coefficient `0.001156` (raises CT win probability)
- `lag_11__T_burning_players`: coefficient `-0.001079` (lowers CT win probability)
- `lag_08__T4__duck_amount`: coefficient `-0.001043` (lowers CT win probability)
- `lag_12__T_place_ROOF`: coefficient `0.000987` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000983` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44023`, seconds `22.00`, LSTM delta `-0.2400`

Top all feature movements:
- `lag_08__T_place_MINI`: contribution `-0.028170`
- `lag_07__CT_place_HUT`: contribution `-0.016126`
- `lag_12__CT_place_HUT`: contribution `-0.015415`
- `lag_10__T_place_HUT`: contribution `-0.011033`
- `lag_11__T5__flash_duration`: contribution `-0.008697`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `-0.008697`
- `lag_14__CT4__flash_duration`: contribution `-0.005383`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.002248`
- `lag_01__CT_B_site_active_infernos`: contribution `-0.002139`

### tick `44343`, seconds `27.00`, LSTM delta `-0.1405`

Top all feature movements:
- `lag_06__CT_flash_duration_sum`: contribution `-0.008235`
- `lag_14__CT_place_HELL`: contribution `-0.007317`
- `lag_06__CT4__flash_duration`: contribution `-0.005804`
- `lag_06__CT_flashed_players`: contribution `-0.005287`
- `lag_06__CT2__flash_duration`: contribution `-0.004586`

Top utility-only movements:
- `lag_06__CT_flash_duration_sum`: contribution `-0.008235`
- `lag_06__CT4__flash_duration`: contribution `-0.005804`
- `lag_06__CT2__flash_duration`: contribution `-0.004586`
- `lag_06__CT1__flash_duration`: contribution `-0.003306`
- `lag_00__CT1__flash_duration`: contribution `-0.003233`

### tick `44055`, seconds `22.50`, LSTM delta `-0.1253`

Top all feature movements:
- `lag_13__CT_place_HUT`: contribution `-0.014730`
- `lag_10__T_place_MINI`: contribution `-0.012971`
- `lag_08__CT_place_HUT`: contribution `-0.011273`
- `lag_09__T_place_MINI`: contribution `+0.007793`
- `lag_11__T_place_HUT`: contribution `-0.007352`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.004117`
- `lag_12__T5__flash_duration`: contribution `-0.002850`
- `lag_11__T_A_site_active_infernos`: contribution `-0.002230`
- `lag_11__T_B_site_active_infernos`: contribution `-0.002025`

### tick `43767`, seconds `18.00`, LSTM delta `+0.0826`

Top all feature movements:
- `lag_01__T_place_MINI`: contribution `+0.012525`
- `lag_00__T_place_MINI`: contribution `+0.006602`
- `lag_12__T_place_ROOF`: contribution `+0.005590`
- `lag_06__CT4__flash_duration`: contribution `+0.004631`
- `lag_04__CT_place_HUT`: contribution `+0.004316`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.004631`
- `lag_15__CT4__flash_duration`: contribution `+0.003569`
- `lag_03__T5__flash_duration`: contribution `+0.003323`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002257`

### tick `43799`, seconds `18.50`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_01__T_place_MINI`: contribution `-0.012525`
- `lag_02__T_place_MINI`: contribution `-0.004988`
- `lag_00__CT_place_HUT`: contribution `+0.004408`
- `lag_00__T_kills_last_3s`: contribution `-0.003114`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002319`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.001916`
