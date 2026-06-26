# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `6`

## Largest probability jumps

- tick `55070`, seconds `70.50`, LSTM `0.5287`, delta `-0.2017`
- tick `54750`, seconds `65.50`, LSTM `0.6877`, delta `+0.1986`
- tick `55230`, seconds `73.00`, LSTM `0.6999`, delta `+0.1785`
- tick `54302`, seconds `58.50`, LSTM `0.7142`, delta `+0.1659`
- tick `56286`, seconds `89.50`, LSTM `0.5513`, delta `-0.1535`
- tick `54398`, seconds `60.00`, LSTM `0.6972`, delta `-0.1509`
- tick `54622`, seconds `63.50`, LSTM `0.5187`, delta `-0.1154`
- tick `54366`, seconds `59.50`, LSTM `0.8480`, delta `+0.0902`
- tick `56190`, seconds `88.00`, LSTM `0.7239`, delta `-0.0528`
- tick `55038`, seconds `70.00`, LSTM `0.7304`, delta `-0.0482`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003879`, |coef| `0.003879`
- `lag_00__T_kills_last_3s`: coefficient `-0.003079`, |coef| `0.003079`
- `lag_00__CT2__duck_amount`: coefficient `-0.001983`, |coef| `0.001983`
- `lag_00__CT_burning_players`: coefficient `0.001979`, |coef| `0.001979`
- `lag_00__CT_kills_last_3s`: coefficient `0.001847`, |coef| `0.001847`
- `lag_00__CT2__alive`: coefficient `0.001756`, |coef| `0.001756`
- `lag_03__CT5__is_walking`: coefficient `0.001752`, |coef| `0.001752`
- `lag_00__T_damage_last_5s`: coefficient `-0.001739`, |coef| `0.001739`
- `lag_10__T_place_DUMPSTER`: coefficient `0.001635`, |coef| `0.001635`
- `lag_00__damage_diff_last_5s`: coefficient `0.001605`, |coef| `0.001605`
- `lag_00__CT2__has_defuser`: coefficient `0.001577`, |coef| `0.001577`
- `lag_00__CT2__shots_fired`: coefficient `-0.001566`, |coef| `0.001566`
- `lag_00__CT_velocity_mean`: coefficient `-0.001558`, |coef| `0.001558`
- `lag_11__T_duck_amount_mean`: coefficient `-0.001475`, |coef| `0.001475`
- `lag_00__CT2__has_helmet`: coefficient `0.001458`, |coef| `0.001458`

## Top 10 utility ridge features

- `lag_13__CT2__flash_duration`: coefficient `0.001337` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001310` (raises CT win probability)
- `lag_12__T2__flash_duration`: coefficient `-0.001306` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001195` (raises CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `0.001150` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.001128` (raises CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `0.001058` (raises CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.001042` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001027` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.001008` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003879` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003079` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `-0.001983` (lowers CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.001979` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001847` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.001756` (raises CT win probability)
- `lag_03__CT5__is_walking`: coefficient `0.001752` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001739` (lowers CT win probability)
- `lag_10__T_place_DUMPSTER`: coefficient `0.001635` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001605` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `55070`, seconds `70.50`, LSTM delta `-0.2017`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009754`
- `lag_13__CT2__flash_duration`: contribution `-0.009592`
- `lag_00__kill_diff_last_3s`: contribution `-0.009337`
- `lag_05__CT_place_ENTRANCE`: contribution `-0.007529`
- `lag_12__CT_shots_fired_sum`: contribution `-0.004491`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `-0.009592`
- `lag_13__CT_flash_duration_sum`: contribution `-0.002442`

### tick `54750`, seconds `65.50`, LSTM delta `+0.1986`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `+0.010779`
- `lag_12__T2__flash_duration`: contribution `+0.010487`
- `lag_00__kill_diff_last_3s`: contribution `+0.009337`
- `lag_00__CT2__shots_fired`: contribution `+0.007783`
- `lag_14__T4__flash_duration`: contribution `+0.007575`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.010487`
- `lag_14__T4__flash_duration`: contribution `+0.007575`
- `lag_12__T_flash_duration_sum`: contribution `+0.002342`

### tick `55230`, seconds `73.00`, LSTM delta `+0.1785`

Top all feature movements:
- `lag_03__CT_place_ENTRANCE`: contribution `+0.010800`
- `lag_00__kill_diff_last_3s`: contribution `+0.009337`
- `lag_15__CT_shots_fired_sum`: contribution `+0.007779`
- `lag_00__CT_kills_last_3s`: contribution `+0.005333`
- `lag_10__CT_place_ENTRANCE`: contribution `+0.004761`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `54302`, seconds `58.50`, LSTM delta `+0.1659`

Top all feature movements:
- `lag_10__T_place_DUMPSTER`: contribution `+0.014865`
- `lag_07__T_place_DUMPSTER`: contribution `+0.009781`
- `lag_00__kill_diff_last_3s`: contribution `+0.009337`
- `lag_02__T2__flash_duration`: contribution `+0.009062`
- `lag_00__T4__flash_duration`: contribution `+0.007464`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.009062`
- `lag_00__T4__flash_duration`: contribution `+0.007464`
- `lag_02__CT2__flash_duration`: contribution `+0.007368`
- `lag_02__T4__flash_duration`: contribution `+0.004830`
- `lag_02__T_flash_duration_sum`: contribution `+0.004680`

### tick `56286`, seconds `89.50`, LSTM delta `-0.1535`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009754`
- `lag_00__kill_diff_last_3s`: contribution `-0.009337`
- `lag_00__CT2__duck_amount`: contribution `-0.006600`
- `lag_00__CT_burning_players`: contribution `-0.005082`
- `lag_00__CT2__alive`: contribution `-0.004252`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.002370`
