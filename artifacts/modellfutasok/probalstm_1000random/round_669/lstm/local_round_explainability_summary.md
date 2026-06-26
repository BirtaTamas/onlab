# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `41332`, seconds `91.00`, LSTM `0.0617`, delta `-0.2153`
- tick `40244`, seconds `74.00`, LSTM `0.3469`, delta `+0.2038`
- tick `38068`, seconds `40.00`, LSTM `0.8155`, delta `+0.1927`
- tick `39188`, seconds `57.50`, LSTM `0.6810`, delta `-0.1556`
- tick `40116`, seconds `72.00`, LSTM `0.1185`, delta `-0.1492`
- tick `39284`, seconds `59.00`, LSTM `0.3960`, delta `-0.1202`
- tick `39220`, seconds `58.00`, LSTM `0.5664`, delta `-0.1146`
- tick `37428`, seconds `30.00`, LSTM `0.7052`, delta `+0.1101`
- tick `39380`, seconds `60.50`, LSTM `0.3705`, delta `+0.0839`
- tick `37556`, seconds `32.00`, LSTM `0.6578`, delta `-0.0804`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005047`, |coef| `0.005047`
- `lag_00__T_kills_last_3s`: coefficient `-0.004208`, |coef| `0.004208`
- `lag_01__CT_place_BACKALLEY`: coefficient `-0.004103`, |coef| `0.004103`
- `lag_00__damage_diff_last_5s`: coefficient `0.003917`, |coef| `0.003917`
- `lag_00__CT4__duck_amount`: coefficient `0.003434`, |coef| `0.003434`
- `lag_05__CT_place_UNDERPASS`: coefficient `-0.003077`, |coef| `0.003077`
- `lag_11__CT_place_TOPOFMID`: coefficient `0.003019`, |coef| `0.003019`
- `lag_05__CT_place_BACKALLEY`: coefficient `0.002760`, |coef| `0.002760`
- `lag_13__T_bomb_zone_count`: coefficient `0.002714`, |coef| `0.002714`
- `lag_01__damage_diff_last_5s`: coefficient `0.002601`, |coef| `0.002601`
- `lag_00__T_damage_last_5s`: coefficient `-0.002564`, |coef| `0.002564`
- `lag_13__CT_place_BACKALLEY`: coefficient `0.002469`, |coef| `0.002469`
- `lag_00__CT_kills_last_3s`: coefficient `0.002218`, |coef| `0.002218`
- `lag_00__T2__is_walking`: coefficient `0.002163`, |coef| `0.002163`
- `lag_10__T_bomb_zone_count`: coefficient `0.002138`, |coef| `0.002138`

## Top 10 utility ridge features

- `lag_05__CT2__flash_duration`: coefficient `0.001836` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.001610` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001498` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001323` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001172` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001151` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.001147` (raises CT win probability)
- `lag_10__T4__smoke`: coefficient `0.001122` (raises CT win probability)
- `lag_06__CT_A_site_active_smokes`: coefficient `0.001109` (raises CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `0.001038` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005047` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004208` (lowers CT win probability)
- `lag_01__CT_place_BACKALLEY`: coefficient `-0.004103` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003917` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.003434` (raises CT win probability)
- `lag_05__CT_place_UNDERPASS`: coefficient `-0.003077` (lowers CT win probability)
- `lag_11__CT_place_TOPOFMID`: coefficient `0.003019` (raises CT win probability)
- `lag_05__CT_place_BACKALLEY`: coefficient `0.002760` (raises CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `0.002714` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002601` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `41332`, seconds `91.00`, LSTM delta `-0.2153`

Top all feature movements:
- `lag_13__T_bomb_zone_count`: contribution `-0.015797`
- `lag_00__T_kills_last_3s`: contribution `-0.013332`
- `lag_00__CT4__duck_amount`: contribution `-0.012612`
- `lag_00__kill_diff_last_3s`: contribution `-0.012147`
- `lag_11__CT_place_SHOP`: contribution `-0.009397`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.004182`

### tick `40244`, seconds `74.00`, LSTM delta `+0.2038`

Top all feature movements:
- `lag_05__CT_place_BACKALLEY`: contribution `+0.041383`
- `lag_05__CT_place_UNDERPASS`: contribution `+0.017844`
- `lag_00__kill_diff_last_3s`: contribution `+0.012147`
- `lag_04__CT_place_UNDERPASS`: contribution `+0.010470`
- `lag_00__CT4__duck_amount`: contribution `+0.009107`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38068`, seconds `40.00`, LSTM delta `+0.1927`

Top all feature movements:
- `lag_05__CT2__flash_duration`: contribution `+0.012777`
- `lag_05__CT_flash_duration_sum`: contribution `+0.012642`
- `lag_00__kill_diff_last_3s`: contribution `+0.012147`
- `lag_05__CT_flashed_players`: contribution `+0.008498`
- `lag_00__damage_diff_last_5s`: contribution `+0.008042`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.012777`
- `lag_05__CT_flash_duration_sum`: contribution `+0.012642`
- `lag_05__CT5__flash_duration`: contribution `+0.006000`
- `lag_05__CT1__flash_duration`: contribution `+0.005841`
- `lag_05__T5__flash_duration`: contribution `+0.002920`

### tick `39188`, seconds `57.50`, LSTM delta `-0.1556`

Top all feature movements:
- `lag_11__CT_place_TOPOFMID`: contribution `-0.021908`
- `lag_00__T_kills_last_3s`: contribution `-0.013332`
- `lag_00__kill_diff_last_3s`: contribution `-0.012147`
- `lag_00__damage_diff_last_5s`: contribution `-0.008837`
- `lag_00__CT_place_CATWALK`: contribution `-0.007991`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40116`, seconds `72.00`, LSTM delta `-0.1492`

Top all feature movements:
- `lag_01__CT_place_BACKALLEY`: contribution `-0.061517`
- `lag_00__T_kills_last_3s`: contribution `-0.013332`
- `lag_00__kill_diff_last_3s`: contribution `-0.012147`
- `lag_01__CT_place_UNDERPASS`: contribution `+0.006548`
- `lag_01__damage_diff_last_5s`: contribution `-0.005341`

Top utility-only movements:
- No utility movement among the top local contributors.
