# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `149216`, seconds `108.00`, LSTM `0.1403`, delta `-0.3857`
- tick `149184`, seconds `107.50`, LSTM `0.5260`, delta `-0.1496`
- tick `147168`, seconds `76.00`, LSTM `0.6578`, delta `+0.1416`
- tick `148096`, seconds `90.50`, LSTM `0.5159`, delta `-0.1407`
- tick `148896`, seconds `103.00`, LSTM `0.6898`, delta `+0.0882`
- tick `149248`, seconds `108.50`, LSTM `0.0608`, delta `-0.0795`
- tick `150816`, seconds `133.00`, LSTM `0.0863`, delta `+0.0665`
- tick `151360`, seconds `141.50`, LSTM `0.2911`, delta `+0.0592`
- tick `151264`, seconds `140.00`, LSTM `0.1808`, delta `+0.0528`
- tick `149088`, seconds `106.00`, LSTM `0.6922`, delta `-0.0498`

## Top 15 local ridge features

- `lag_14__CT3__shots_fired`: coefficient `0.003444`, |coef| `0.003444`
- `lag_14__CT_shots_fired_sum`: coefficient `0.002819`, |coef| `0.002819`
- `lag_00__kill_diff_last_3s`: coefficient `0.002777`, |coef| `0.002777`
- `lag_10__CT1__shots_fired`: coefficient `0.002368`, |coef| `0.002368`
- `lag_10__CT_shots_fired_sum`: coefficient `0.002345`, |coef| `0.002345`
- `lag_04__T_place_BDOORS`: coefficient `-0.002281`, |coef| `0.002281`
- `lag_00__damage_diff_last_5s`: coefficient `0.001965`, |coef| `0.001965`
- `lag_00__T_place_HOLE`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_00__CT_kills_last_3s`: coefficient `0.001781`, |coef| `0.001781`
- `lag_05__T_place_BDOORS`: coefficient `-0.001763`, |coef| `0.001763`
- `lag_00__T_kills_last_3s`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_03__T_place_BDOORS`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_08__CT4__is_walking`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_11__T_place_BDOORS`: coefficient `-0.001627`, |coef| `0.001627`
- `lag_12__CT4__is_walking`: coefficient `-0.001597`, |coef| `0.001597`

## Top 10 utility ridge features

- `lag_06__CT2__smoke`: coefficient `-0.001088` (lowers CT win probability)
- `lag_12__T5__flash_duration`: coefficient `0.001065` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.000994` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000850` (lowers CT win probability)
- `lag_04__T_smokes_last_5s`: coefficient `-0.000780` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000756` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.000736` (lowers CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `0.000693` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.000668` (lowers CT win probability)
- `lag_01__active_infernos_total`: coefficient `0.000622` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT3__shots_fired`: coefficient `0.003444` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.002819` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002777` (raises CT win probability)
- `lag_10__CT1__shots_fired`: coefficient `0.002368` (raises CT win probability)
- `lag_10__CT_shots_fired_sum`: coefficient `0.002345` (raises CT win probability)
- `lag_04__T_place_BDOORS`: coefficient `-0.002281` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001965` (raises CT win probability)
- `lag_00__T_place_HOLE`: coefficient `-0.001850` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001781` (raises CT win probability)
- `lag_05__T_place_BDOORS`: coefficient `-0.001763` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `149216`, seconds `108.00`, LSTM delta `-0.3857`

Top all feature movements:
- `lag_14__CT3__shots_fired`: contribution `-0.044274`
- `lag_14__CT_shots_fired_sum`: contribution `-0.043080`
- `lag_10__CT_shots_fired_sum`: contribution `-0.029319`
- `lag_04__T_place_BDOORS`: contribution `-0.028529`
- `lag_10__CT1__shots_fired`: contribution `-0.022523`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149184`, seconds `107.50`, LSTM delta `-0.1496`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `-0.028529`
- `lag_03__T_place_BDOORS`: contribution `-0.020906`
- `lag_13__CT3__shots_fired`: contribution `-0.019103`
- `lag_14__CT_shots_fired_sum`: contribution `+0.013707`
- `lag_14__CT3__shots_fired`: contribution `+0.012397`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `147168`, seconds `76.00`, LSTM delta `+0.1416`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.006683`
- `lag_13__CT2__duck_amount`: contribution `+0.005417`
- `lag_00__CT_kills_last_3s`: contribution `+0.005142`
- `lag_08__CT2__duck_amount`: contribution `+0.004918`
- `lag_00__damage_diff_last_5s`: contribution `+0.004432`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `148096`, seconds `90.50`, LSTM delta `-0.1407`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.011023`
- `lag_12__T5__flash_duration`: contribution `-0.007316`
- `lag_00__kill_diff_last_3s`: contribution `-0.006683`
- `lag_08__CT3__flash_duration`: contribution `-0.006641`
- `lag_10__CT_place_BDOORS`: contribution `-0.006096`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.007316`
- `lag_08__CT3__flash_duration`: contribution `-0.006641`
- `lag_12__CT4__flash_duration`: contribution `-0.004391`
- `lag_14__CT1__flash_duration`: contribution `-0.003474`

### tick `148896`, seconds `103.00`, LSTM delta `+0.0882`

Top all feature movements:
- `lag_00__CT1__shots_fired`: contribution `+0.009342`
- `lag_04__CT3__shots_fired`: contribution `+0.008932`
- `lag_00__kill_diff_last_3s`: contribution `+0.006683`
- `lag_00__bomb_events_last_5s`: contribution `+0.005836`
- `lag_00__CT_kills_last_3s`: contribution `+0.005142`

Top utility-only movements:
- No utility movement among the top local contributors.
