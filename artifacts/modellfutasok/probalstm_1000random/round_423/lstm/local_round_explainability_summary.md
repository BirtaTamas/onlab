# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `125522`, seconds `36.00`, LSTM `0.8570`, delta `+0.2615`
- tick `127762`, seconds `71.00`, LSTM `0.9373`, delta `+0.2192`
- tick `124338`, seconds `17.50`, LSTM `0.5024`, delta `-0.1917`
- tick `124018`, seconds `12.50`, LSTM `0.6962`, delta `+0.1263`
- tick `127570`, seconds `68.00`, LSTM `0.8332`, delta `-0.1125`
- tick `124114`, seconds `14.00`, LSTM `0.8460`, delta `+0.1090`
- tick `124146`, seconds `14.50`, LSTM `0.7375`, delta `-0.1085`
- tick `125586`, seconds `37.00`, LSTM `0.9410`, delta `+0.0816`
- tick `124690`, seconds `23.00`, LSTM `0.5445`, delta `-0.0632`
- tick `124658`, seconds `22.50`, LSTM `0.6076`, delta `+0.0545`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004210`, |coef| `0.004210`
- `lag_00__CT_kills_last_3s`: coefficient `0.003313`, |coef| `0.003313`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002699`, |coef| `0.002699`
- `lag_00__T_place_LONGA`: coefficient `0.002336`, |coef| `0.002336`
- `lag_04__CT1__duck_amount`: coefficient `0.002206`, |coef| `0.002206`
- `lag_00__damage_diff_last_5s`: coefficient `0.002196`, |coef| `0.002196`
- `lag_14__T1__duck_amount`: coefficient `-0.002059`, |coef| `0.002059`
- `lag_10__T_place_CATWALK`: coefficient `-0.002047`, |coef| `0.002047`
- `lag_00__T_kills_last_3s`: coefficient `-0.001906`, |coef| `0.001906`
- `lag_00__T1__alive`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_07__CT_place_EXTENDEDA`: coefficient `-0.001852`, |coef| `0.001852`
- `lag_00__T_place_LONGDOORS`: coefficient `-0.001819`, |coef| `0.001819`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001814`, |coef| `0.001814`
- `lag_07__CT_place_SHORTSTAIRS`: coefficient `0.001768`, |coef| `0.001768`
- `lag_04__CT5__is_walking`: coefficient `0.001727`, |coef| `0.001727`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001814` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001674` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.001408` (lowers CT win probability)
- `lag_11__CT1__flash`: coefficient `0.001402` (raises CT win probability)
- `lag_11__T_active_smokes`: coefficient `-0.001392` (lowers CT win probability)
- `lag_11__active_smokes_total`: coefficient `-0.001370` (lowers CT win probability)
- `lag_13__T_active_smokes`: coefficient `-0.001288` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.001235` (raises CT win probability)
- `lag_10__T_he_last_5s`: coefficient `-0.001163` (lowers CT win probability)
- `lag_07__CT_active_smokes`: coefficient `-0.001150` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004210` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003313` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002699` (raises CT win probability)
- `lag_00__T_place_LONGA`: coefficient `0.002336` (raises CT win probability)
- `lag_04__CT1__duck_amount`: coefficient `0.002206` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002196` (raises CT win probability)
- `lag_14__T1__duck_amount`: coefficient `-0.002059` (lowers CT win probability)
- `lag_10__T_place_CATWALK`: coefficient `-0.002047` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001906` (lowers CT win probability)
- `lag_00__T1__alive`: coefficient `-0.001878` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `125522`, seconds `36.00`, LSTM delta `+0.2615`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.010133`
- `lag_00__T_place_LONGA`: contribution `+0.009952`
- `lag_00__CT_kills_last_3s`: contribution `+0.009565`
- `lag_04__CT1__duck_amount`: contribution `+0.008415`
- `lag_14__T1__duck_amount`: contribution `+0.008061`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127762`, seconds `71.00`, LSTM delta `+0.2192`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.020267`
- `lag_00__T_flash_alpha_mean`: contribution `+0.011007`
- `lag_07__CT_place_EXTENDEDA`: contribution `+0.010399`
- `lag_07__CT_place_SHORTSTAIRS`: contribution `+0.009857`
- `lag_00__CT_kills_last_3s`: contribution `+0.009565`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.011007`
- `lag_12__T4__flash_duration`: contribution `+0.008366`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.003495`

### tick `124338`, seconds `17.50`, LSTM delta `-0.1917`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `-0.010976`
- `lag_06__T_shots_fired_sum`: contribution `-0.009387`
- `lag_04__CT1__duck_amount`: contribution `-0.008415`
- `lag_07__T_shots_fired_sum`: contribution `-0.007813`
- `lag_00__damage_diff_last_5s`: contribution `-0.006095`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `-0.004446`
- `lag_07__T_flash_duration_sum`: contribution `-0.004226`
- `lag_07__T2__flash_duration`: contribution `-0.004020`
- `lag_07__T3__flash_duration`: contribution `-0.003940`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002508`

### tick `124018`, seconds `12.50`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `+0.015177`
- `lag_07__CT_place_HOLE`: contribution `-0.010335`
- `lag_00__kill_diff_last_3s`: contribution `+0.010133`
- `lag_00__CT_kills_last_3s`: contribution `+0.009565`
- `lag_04__CT_place_HOLE`: contribution `+0.009040`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `+0.015177`
- `lag_07__T3__flash_duration`: contribution `+0.003463`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002508`
- `lag_10__CT1__flash_duration`: contribution `+0.001972`

### tick `127570`, seconds `68.00`, LSTM delta `-0.1125`

Top all feature movements:
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.010399`
- `lag_00__kill_diff_last_3s`: contribution `-0.010133`
- `lag_07__CT_place_SHORTSTAIRS`: contribution `-0.009857`
- `lag_06__T4__flash_duration`: contribution `-0.007342`
- `lag_00__T_kills_last_3s`: contribution `-0.006039`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.007342`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.003671`
- `lag_11__T_A_site_active_infernos`: contribution `-0.002044`
