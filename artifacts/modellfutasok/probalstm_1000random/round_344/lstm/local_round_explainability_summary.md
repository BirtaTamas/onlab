# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `131052`, seconds `61.50`, LSTM `0.4633`, delta `+0.3248`
- tick `131084`, seconds `62.00`, LSTM `0.2381`, delta `-0.2252`
- tick `131116`, seconds `62.50`, LSTM `0.0909`, delta `-0.1472`
- tick `130988`, seconds `60.50`, LSTM `0.2254`, delta `-0.1066`
- tick `128460`, seconds `21.00`, LSTM `0.1642`, delta `+0.0941`
- tick `131020`, seconds `61.00`, LSTM `0.1385`, delta `-0.0868`
- tick `130828`, seconds `58.00`, LSTM `0.3770`, delta `-0.0553`
- tick `129228`, seconds `33.00`, LSTM `0.1866`, delta `-0.0479`
- tick `130060`, seconds `46.00`, LSTM `0.3556`, delta `+0.0442`
- tick `127148`, seconds `0.50`, LSTM `0.0310`, delta `-0.0423`

## Top 15 local ridge features

- `lag_01__T4__flash_duration`: coefficient `0.003250`, |coef| `0.003250`
- `lag_01__T_flashed_players`: coefficient `0.003033`, |coef| `0.003033`
- `lag_00__T4__flash_duration`: coefficient `-0.002473`, |coef| `0.002473`
- `lag_02__CT_place_PIT`: coefficient `0.002406`, |coef| `0.002406`
- `lag_00__CT_place_PIT`: coefficient `-0.002383`, |coef| `0.002383`
- `lag_08__CT4__is_walking`: coefficient `0.002344`, |coef| `0.002344`
- `lag_06__T_flashed_players`: coefficient `0.002307`, |coef| `0.002307`
- `lag_06__CT_flashed_players`: coefficient `0.002235`, |coef| `0.002235`
- `lag_00__T1__shots_fired`: coefficient `-0.002207`, |coef| `0.002207`
- `lag_00__kill_diff_last_3s`: coefficient `0.002204`, |coef| `0.002204`
- `lag_06__CT1__flash_duration`: coefficient `0.002037`, |coef| `0.002037`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001943`, |coef| `0.001943`
- `lag_01__T_flash_duration_sum`: coefficient `0.001917`, |coef| `0.001917`
- `lag_00__damage_diff_last_5s`: coefficient `0.001894`, |coef| `0.001894`
- `lag_04__CT_flashed_players`: coefficient `-0.001874`, |coef| `0.001874`

## Top 10 utility ridge features

- `lag_01__T4__flash_duration`: coefficient `0.003250` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.002473` (lowers CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.002037` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.001917` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.001339` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.001312` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.001187` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.001132` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.001024` (lowers CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000973` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_flashed_players`: coefficient `0.003033` (raises CT win probability)
- `lag_02__CT_place_PIT`: coefficient `0.002406` (raises CT win probability)
- `lag_00__CT_place_PIT`: coefficient `-0.002383` (lowers CT win probability)
- `lag_08__CT4__is_walking`: coefficient `0.002344` (raises CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.002307` (raises CT win probability)
- `lag_06__CT_flashed_players`: coefficient `0.002235` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.002207` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002204` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001943` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001894` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `131052`, seconds `61.50`, LSTM delta `+0.3248`

Top all feature movements:
- `lag_01__T_flashed_players`: contribution `+0.017560`
- `lag_00__T_shots_fired_sum`: contribution `+0.017483`
- `lag_01__T4__flash_duration`: contribution `+0.017381`
- `lag_00__T1__shots_fired`: contribution `+0.015831`
- `lag_06__CT_flashed_players`: contribution `+0.014686`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.017381`
- `lag_00__T4__flash_duration`: contribution `+0.013224`
- `lag_06__CT1__flash_duration`: contribution `+0.011274`
- `lag_01__T_flash_duration_sum`: contribution `+0.006511`
- `lag_01__CT1__flash_duration`: contribution `+0.004954`

### tick `131084`, seconds `62.00`, LSTM delta `-0.2252`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.017483`
- `lag_01__T4__flash_duration`: contribution `-0.017381`
- `lag_00__T1__shots_fired`: contribution `-0.009235`
- `lag_06__T_flashed_players`: contribution `-0.008905`
- `lag_01__T1__shots_fired`: contribution `-0.008135`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.017381`
- `lag_07__CT1__flash_duration`: contribution `-0.007262`
- `lag_01__T_flash_duration_sum`: contribution `-0.004168`

### tick `131116`, seconds `62.50`, LSTM delta `-0.1472`

Top all feature movements:
- `lag_00__T1__shots_fired`: contribution `+0.010554`
- `lag_02__CT_place_PIT`: contribution `-0.010359`
- `lag_08__CT_flashed_players`: contribution `-0.008565`
- `lag_02__T_shots_fired_sum`: contribution `-0.007236`
- `lag_01__T_flashed_players`: contribution `-0.005853`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `-0.005302`
- `lag_08__CT_flash_duration_sum`: contribution `-0.003382`
- `lag_03__T4__flash_duration`: contribution `-0.003212`

### tick `130988`, seconds `60.50`, LSTM delta `-0.1066`

Top all feature movements:
- `lag_04__CT_flashed_players`: contribution `-0.012311`
- `lag_00__CT_place_PIT`: contribution `-0.010262`
- `lag_00__T_shots_fired_sum`: contribution `-0.008742`
- `lag_00__T1__shots_fired`: contribution `-0.007915`
- `lag_00__T1__duck_amount`: contribution `-0.006349`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `-0.004408`
- `lag_04__CT_flash_duration_sum`: contribution `-0.003407`
- `lag_10__T_B_site_active_infernos`: contribution `-0.002574`

### tick `128460`, seconds `21.00`, LSTM delta `+0.0941`

Top all feature movements:
- `lag_11__CT5__flash_duration`: contribution `+0.007202`
- `lag_00__kill_diff_last_3s`: contribution `+0.005306`
- `lag_00__CT_kills_last_3s`: contribution `+0.003922`
- `lag_06__T5__duck_amount`: contribution `+0.003692`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003074`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.007202`
- `lag_11__CT_flash_duration_sum`: contribution `+0.002334`
