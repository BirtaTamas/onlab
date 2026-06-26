# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `13881`, seconds `18.50`, LSTM `0.3107`, delta `+0.1711`
- tick `13497`, seconds `12.50`, LSTM `0.0792`, delta `-0.1208`
- tick `16985`, seconds `67.00`, LSTM `0.0199`, delta `-0.0963`
- tick `16889`, seconds `65.50`, LSTM `0.1606`, delta `+0.0896`
- tick `14105`, seconds `22.00`, LSTM `0.4199`, delta `+0.0749`
- tick `16601`, seconds `61.00`, LSTM `0.1678`, delta `-0.0736`
- tick `13913`, seconds `19.00`, LSTM `0.3743`, delta `+0.0636`
- tick `16537`, seconds `60.00`, LSTM `0.2771`, delta `-0.0601`
- tick `13209`, seconds `8.00`, LSTM `0.2468`, delta `-0.0549`
- tick `16697`, seconds `62.50`, LSTM `0.0803`, delta `-0.0485`

## Top 15 local ridge features

- `lag_13__T_place_RAMP`: coefficient `-0.001617`, |coef| `0.001617`
- `lag_15__T_place_TSIDELOWER`: coefficient `0.001613`, |coef| `0.001613`
- `lag_06__CT_place_HOUSE`: coefficient `0.001547`, |coef| `0.001547`
- `lag_14__T_place_RAMP`: coefficient `-0.001543`, |coef| `0.001543`
- `lag_05__T3__flash_duration`: coefficient `-0.001456`, |coef| `0.001456`
- `lag_06__T_shots_fired_sum`: coefficient `-0.001414`, |coef| `0.001414`
- `lag_00__kill_diff_last_3s`: coefficient `0.001364`, |coef| `0.001364`
- `lag_12__T_place_RAMP`: coefficient `-0.001339`, |coef| `0.001339`
- `lag_15__T_place_RAMP`: coefficient `-0.001332`, |coef| `0.001332`
- `lag_06__T3__flash_duration`: coefficient `-0.001300`, |coef| `0.001300`
- `lag_00__damage_diff_last_5s`: coefficient `0.001255`, |coef| `0.001255`
- `lag_11__T_place_RAMP`: coefficient `-0.001204`, |coef| `0.001204`
- `lag_10__T_place_RAMP`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_10__T_place_TSIDELOWER`: coefficient `0.001171`, |coef| `0.001171`
- `lag_12__T_place_TUNNEL`: coefficient `-0.001166`, |coef| `0.001166`

## Top 10 utility ridge features

- `lag_05__T3__flash_duration`: coefficient `-0.001456` (lowers CT win probability)
- `lag_06__T3__flash_duration`: coefficient `-0.001300` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.001066` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.000990` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000891` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000866` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000830` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.000766` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000763` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000762` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_RAMP`: coefficient `-0.001617` (lowers CT win probability)
- `lag_15__T_place_TSIDELOWER`: coefficient `0.001613` (raises CT win probability)
- `lag_06__CT_place_HOUSE`: coefficient `0.001547` (raises CT win probability)
- `lag_14__T_place_RAMP`: coefficient `-0.001543` (lowers CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `-0.001414` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001364` (raises CT win probability)
- `lag_12__T_place_RAMP`: coefficient `-0.001339` (lowers CT win probability)
- `lag_15__T_place_RAMP`: coefficient `-0.001332` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001255` (raises CT win probability)
- `lag_11__T_place_RAMP`: coefficient `-0.001204` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `13881`, seconds `18.50`, LSTM delta `+0.1711`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.013786`
- `lag_05__T3__flash_duration`: contribution `+0.009935`
- `lag_12__CT3__flash_duration`: contribution `+0.006650`
- `lag_06__T2__shots_fired`: contribution `+0.006577`
- `lag_04__T1__flash_duration`: contribution `+0.006258`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `+0.009935`
- `lag_12__CT3__flash_duration`: contribution `+0.006650`
- `lag_04__T1__flash_duration`: contribution `+0.006258`
- `lag_00__CT4__flash_duration`: contribution `+0.004341`
- `lag_15__T1__flash_duration`: contribution `+0.004061`

### tick `13497`, seconds `12.50`, LSTM delta `-0.1208`

Top all feature movements:
- `lag_06__T3__flash_duration`: contribution `-0.008869`
- `lag_09__CT_place_TOPOFMID`: contribution `-0.005880`
- `lag_06__CT4__flash_duration`: contribution `-0.004397`
- `lag_06__CT_flashed_players`: contribution `-0.004360`
- `lag_06__T_flashed_players`: contribution `-0.004235`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `-0.008869`
- `lag_06__CT4__flash_duration`: contribution `-0.004397`
- `lag_00__CT3__flash_duration`: contribution `-0.003884`
- `lag_06__T_flash_duration_sum`: contribution `-0.003569`
- `lag_03__T1__flash_duration`: contribution `-0.003490`

### tick `16985`, seconds `67.00`, LSTM delta `-0.0963`

Top all feature movements:
- `lag_12__T4__shots_fired`: contribution `-0.008171`
- `lag_13__T_shots_fired_sum`: contribution `-0.004642`
- `lag_00__damage_diff_last_5s`: contribution `-0.004588`
- `lag_10__T_place_TSIDELOWER`: contribution `-0.004391`
- `lag_03__CT_flashed_players`: contribution `-0.004347`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.003831`
- `lag_03__CT_flash_duration_sum`: contribution `-0.003494`

### tick `16889`, seconds `65.50`, LSTM delta `+0.0896`

Top all feature movements:
- `lag_09__T4__shots_fired`: contribution `+0.012038`
- `lag_09__T_shots_fired_sum`: contribution `+0.009049`
- `lag_13__T_place_RAMP`: contribution `+0.005719`
- `lag_14__T_place_RAMP`: contribution `+0.005459`
- `lag_00__CT4__flash_duration`: contribution `-0.005280`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.005280`
- `lag_00__CT2__flash_duration`: contribution `+0.001489`

### tick `14105`, seconds `22.00`, LSTM delta `+0.0749`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `+0.010057`
- `lag_13__T2__shots_fired`: contribution `+0.004109`
- `lag_00__CT_place_SIDEHALL`: contribution `+0.003432`
- `lag_07__CT4__flash_duration`: contribution `+0.003116`
- `lag_12__T3__flash_duration`: contribution `+0.003078`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.003116`
- `lag_12__T3__flash_duration`: contribution `+0.003078`
- `lag_11__T1__flash_duration`: contribution `+0.002621`
