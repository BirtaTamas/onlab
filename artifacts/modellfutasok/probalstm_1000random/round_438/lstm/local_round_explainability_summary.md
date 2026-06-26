# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `12`

## Largest probability jumps

- tick `101737`, seconds `99.50`, LSTM `0.7075`, delta `+0.1634`
- tick `102345`, seconds `109.00`, LSTM `0.7982`, delta `-0.1298`
- tick `96393`, seconds `16.00`, LSTM `0.5306`, delta `+0.1112`
- tick `102153`, seconds `106.00`, LSTM `0.9327`, delta `+0.0997`
- tick `101769`, seconds `100.00`, LSTM `0.7704`, delta `+0.0630`
- tick `101929`, seconds `102.50`, LSTM `0.8134`, delta `-0.0613`
- tick `101993`, seconds `103.50`, LSTM `0.8542`, delta `+0.0583`
- tick `102633`, seconds `113.50`, LSTM `0.8063`, delta `-0.0576`
- tick `96425`, seconds `16.50`, LSTM `0.5766`, delta `+0.0460`
- tick `97353`, seconds `31.00`, LSTM `0.5913`, delta `-0.0392`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.002305`, |coef| `0.002305`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001640`, |coef| `0.001640`
- `lag_00__CT_kills_last_3s`: coefficient `0.001639`, |coef| `0.001639`
- `lag_00__kill_diff_last_3s`: coefficient `0.001535`, |coef| `0.001535`
- `lag_06__T_place_OUTSIDELONG`: coefficient `0.001382`, |coef| `0.001382`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001308`, |coef| `0.001308`
- `lag_07__CT_place_BRICKS`: coefficient `0.001294`, |coef| `0.001294`
- `lag_06__CT_A_site_active_infernos`: coefficient `0.001249`, |coef| `0.001249`
- `lag_03__T_B_site_active_smokes`: coefficient `0.001185`, |coef| `0.001185`
- `lag_00__CT3__duck_amount`: coefficient `0.001180`, |coef| `0.001180`
- `lag_00__T1__molly`: coefficient `-0.001165`, |coef| `0.001165`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001118`, |coef| `0.001118`
- `lag_04__T_place_OUTSIDELONG`: coefficient `0.001095`, |coef| `0.001095`
- `lag_03__CT3__duck_amount`: coefficient `-0.001073`, |coef| `0.001073`
- `lag_00__T1__alive`: coefficient `-0.001067`, |coef| `0.001067`

## Top 10 utility ridge features

- `lag_06__CT_A_site_active_infernos`: coefficient `0.001249` (raises CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `0.001185` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.001165` (lowers CT win probability)
- `lag_02__T_B_site_active_smokes`: coefficient `0.001029` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.000988` (raises CT win probability)
- `lag_10__CT4__molly`: coefficient `-0.000893` (lowers CT win probability)
- `lag_15__T4__smoke`: coefficient `-0.000784` (lowers CT win probability)
- `lag_14__T3__smoke`: coefficient `-0.000784` (lowers CT win probability)
- `lag_04__T_B_site_active_smokes`: coefficient `0.000777` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000762` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.002305` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001640` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001639` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001535` (raises CT win probability)
- `lag_06__T_place_OUTSIDELONG`: coefficient `0.001382` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001308` (raises CT win probability)
- `lag_07__CT_place_BRICKS`: coefficient `0.001294` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001180` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001118` (raises CT win probability)
- `lag_04__T_place_OUTSIDELONG`: coefficient `0.001095` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `101737`, seconds `99.50`, LSTM delta `+0.1634`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004732`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.004407`
- `lag_00__CT3__duck_amount`: contribution `+0.004392`
- `lag_03__CT3__duck_amount`: contribution `+0.003993`
- `lag_00__kill_diff_last_3s`: contribution `+0.003696`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `+0.004407`
- `lag_02__T_B_site_active_infernos`: contribution `+0.002793`
- `lag_00__T1__molly`: contribution `+0.002580`
- `lag_10__CT4__molly`: contribution `+0.002200`

### tick `102345`, seconds `109.00`, LSTM delta `-0.1298`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011087`
- `lag_15__CT_shots_fired_sum`: contribution `-0.009229`
- `lag_14__T_shots_fired_sum`: contribution `-0.006935`
- `lag_02__CT_shots_fired_sum`: contribution `-0.006362`
- `lag_11__CT_place_BRICKS`: contribution `-0.005802`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.003700`

### tick `96393`, seconds `16.00`, LSTM delta `+0.1112`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `+0.011553`
- `lag_14__CT_place_BRIDGE`: contribution `+0.005991`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005695`
- `lag_00__CT_kills_last_3s`: contribution `+0.004732`
- `lag_15__T_place_TSTAIRS`: contribution `+0.004505`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.003763`
- `lag_02__T_B_site_active_smokes`: contribution `+0.001559`

### tick `102153`, seconds `106.00`, LSTM delta `+0.0997`

Top all feature movements:
- `lag_07__CT_place_BRICKS`: contribution `+0.024843`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005695`
- `lag_00__CT_kills_last_3s`: contribution `+0.004732`
- `lag_09__CT_shots_fired_sum`: contribution `+0.003881`
- `lag_00__kill_diff_last_3s`: contribution `+0.003696`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101769`, seconds `100.00`, LSTM delta `+0.0630`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.005695`
- `lag_07__CT3__duck_amount`: contribution `+0.003139`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.002691`
- `lag_01__CT3__duck_amount`: contribution `+0.002674`
- `lag_00__CT_place_WALKWAY`: contribution `-0.002672`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.002691`
- `lag_03__T_B_site_active_infernos`: contribution `+0.001875`
- `lag_03__T_B_site_active_smokes`: contribution `+0.001795`
- `lag_01__T1__molly`: contribution `+0.001627`
