# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `16706`, seconds `82.00`, LSTM `0.8141`, delta `+0.2831`
- tick `17922`, seconds `101.00`, LSTM `0.6186`, delta `-0.2602`
- tick `16674`, seconds `81.50`, LSTM `0.5310`, delta `+0.2526`
- tick `17858`, seconds `100.00`, LSTM `0.7130`, delta `-0.2140`
- tick `17890`, seconds `100.50`, LSTM `0.8789`, delta `+0.1658`
- tick `16770`, seconds `83.00`, LSTM `0.9355`, delta `+0.1136`
- tick `11490`, seconds `0.50`, LSTM `0.2218`, delta `-0.0635`
- tick `18050`, seconds `103.00`, LSTM `0.6799`, delta `+0.0469`
- tick `11842`, seconds `6.00`, LSTM `0.2543`, delta `+0.0435`
- tick `16642`, seconds `81.00`, LSTM `0.2785`, delta `+0.0398`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004931`, |coef| `0.004931`
- `lag_00__T_shots_fired_sum`: coefficient `-0.004547`, |coef| `0.004547`
- `lag_01__CT_shots_fired_sum`: coefficient `0.004185`, |coef| `0.004185`
- `lag_00__damage_diff_last_5s`: coefficient `0.004006`, |coef| `0.004006`
- `lag_00__CT3__shots_fired`: coefficient `0.003883`, |coef| `0.003883`
- `lag_00__CT_kills_last_3s`: coefficient `0.003753`, |coef| `0.003753`
- `lag_01__T3__shots_fired`: coefficient `0.003588`, |coef| `0.003588`
- `lag_00__bomb_events_last_5s`: coefficient `0.002908`, |coef| `0.002908`
- `lag_00__CT_place_RUINS`: coefficient `0.002653`, |coef| `0.002653`
- `lag_00__T3__shots_fired`: coefficient `0.002619`, |coef| `0.002619`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002507`, |coef| `0.002507`
- `lag_00__CT4__shots_fired`: coefficient `-0.002462`, |coef| `0.002462`
- `lag_00__T_kills_last_3s`: coefficient `-0.002372`, |coef| `0.002372`
- `lag_00__CT_damage_last_5s`: coefficient `0.002321`, |coef| `0.002321`
- `lag_02__T3__shots_fired`: coefficient `0.002289`, |coef| `0.002289`

## Top 10 utility ridge features

- `lag_06__CT1__smoke`: coefficient `-0.001658` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.001540` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.001401` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.001347` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.001346` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001268` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `0.001242` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `-0.001205` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.001198` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.001170` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004931` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.004547` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.004185` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004006` (raises CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `0.003883` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003753` (raises CT win probability)
- `lag_01__T3__shots_fired`: coefficient `0.003588` (raises CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.002908` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.002653` (raises CT win probability)
- `lag_00__T3__shots_fired`: coefficient `0.002619` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `16706`, seconds `82.00`, LSTM delta `+0.2831`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.020355`
- `lag_00__CT3__shots_fired`: contribution `+0.013979`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013936`
- `lag_00__kill_diff_last_3s`: contribution `+0.011868`
- `lag_00__CT_kills_last_3s`: contribution `+0.010834`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17922`, seconds `101.00`, LSTM delta `-0.2602`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.040709`
- `lag_01__T_shots_fired_sum`: contribution `-0.020605`
- `lag_00__kill_diff_last_3s`: contribution `-0.011868`
- `lag_01__CT4__shots_fired`: contribution `-0.010281`
- `lag_02__CT_shots_fired_sum`: contribution `+0.007909`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `-0.004294`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.004228`

### tick `16674`, seconds `81.50`, LSTM delta `+0.2526`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.020355`
- `lag_00__T_shots_fired_sum`: contribution `+0.017045`
- `lag_00__CT3__shots_fired`: contribution `+0.013979`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012194`
- `lag_00__bomb_events_last_5s`: contribution `+0.012153`

Top utility-only movements:
- `lag_06__CT1__smoke`: contribution `+0.003595`

### tick `17858`, seconds `100.00`, LSTM delta `-0.2140`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.034090`
- `lag_01__CT_shots_fired_sum`: contribution `+0.017447`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012194`
- `lag_00__damage_diff_last_5s`: contribution `-0.011930`
- `lag_00__kill_diff_last_3s`: contribution `-0.011868`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `-0.004943`
- `lag_01__CT3__flash_duration`: contribution `-0.004908`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.004750`

### tick `17890`, seconds `100.50`, LSTM delta `+0.1658`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.047726`
- `lag_00__CT_shots_fired_sum`: contribution `-0.024388`
- `lag_01__CT_shots_fired_sum`: contribution `+0.020355`
- `lag_00__CT4__shots_fired`: contribution `+0.018569`
- `lag_01__T_shots_fired_sum`: contribution `+0.014718`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.004908`
- `lag_02__CT3__flash_duration`: contribution `+0.004294`
