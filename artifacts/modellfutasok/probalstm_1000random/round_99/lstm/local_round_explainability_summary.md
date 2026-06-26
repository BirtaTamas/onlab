# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `67395`, seconds `21.50`, LSTM `0.5184`, delta `+0.2849`
- tick `68035`, seconds `31.50`, LSTM `0.0794`, delta `-0.2169`
- tick `67331`, seconds `20.50`, LSTM `0.2711`, delta `-0.2103`
- tick `66915`, seconds `14.00`, LSTM `0.5418`, delta `+0.1362`
- tick `67011`, seconds `15.50`, LSTM `0.4685`, delta `-0.0854`
- tick `67715`, seconds `26.50`, LSTM `0.3753`, delta `-0.0492`
- tick `66723`, seconds `11.00`, LSTM `0.3747`, delta `+0.0461`
- tick `67939`, seconds `30.00`, LSTM `0.3329`, delta `+0.0433`
- tick `66531`, seconds `8.00`, LSTM `0.3021`, delta `-0.0430`
- tick `66467`, seconds `7.00`, LSTM `0.3420`, delta `-0.0417`

## Top 15 local ridge features

- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001987`, |coef| `0.001987`
- `lag_08__T_flashes_last_5s`: coefficient `-0.001834`, |coef| `0.001834`
- `lag_00__kill_diff_last_3s`: coefficient `0.001749`, |coef| `0.001749`
- `lag_00__T2__flash_duration`: coefficient `-0.001743`, |coef| `0.001743`
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `-0.001703`, |coef| `0.001703`
- `lag_00__CT4__is_scoped`: coefficient `-0.001684`, |coef| `0.001684`
- `lag_08__CT_place_LOWERTUNNEL`: coefficient `-0.001644`, |coef| `0.001644`
- `lag_08__CT_place_CATWALK`: coefficient `0.001519`, |coef| `0.001519`
- `lag_10__T_place_TUNNELSTAIRS`: coefficient `0.001358`, |coef| `0.001358`
- `lag_00__T_kills_last_3s`: coefficient `-0.001310`, |coef| `0.001310`
- `lag_02__T2__duck_amount`: coefficient `-0.001244`, |coef| `0.001244`
- `lag_15__CT4__is_scoped`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_02__T2__shots_fired`: coefficient `-0.001230`, |coef| `0.001230`
- `lag_12__CT_place_TOPOFMID`: coefficient `-0.001225`, |coef| `0.001225`
- `lag_02__CT_place_LOWERTUNNEL`: coefficient `-0.001211`, |coef| `0.001211`

## Top 10 utility ridge features

- `lag_08__T_flashes_last_5s`: coefficient `-0.001834` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001743` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.000986` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000937` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000801` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `0.000794` (raises CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.000761` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `0.000746` (raises CT win probability)
- `lag_14__CT_B_site_active_infernos`: coefficient `-0.000655` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `-0.000636` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001987` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001749` (raises CT win probability)
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `-0.001703` (lowers CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001684` (lowers CT win probability)
- `lag_08__CT_place_LOWERTUNNEL`: coefficient `-0.001644` (lowers CT win probability)
- `lag_08__CT_place_CATWALK`: coefficient `0.001519` (raises CT win probability)
- `lag_10__T_place_TUNNELSTAIRS`: coefficient `0.001358` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001310` (lowers CT win probability)
- `lag_02__T2__duck_amount`: coefficient `-0.001244` (lowers CT win probability)
- `lag_15__CT4__is_scoped`: coefficient `-0.001242` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `67395`, seconds `21.50`, LSTM delta `+0.2849`

Top all feature movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.016616`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `+0.009482`
- `lag_00__T2__flash_duration`: contribution `+0.009119`
- `lag_12__CT_place_TOPOFMID`: contribution `+0.008889`
- `lag_02__T2__shots_fired`: contribution `+0.005789`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.016616`
- `lag_00__T2__flash_duration`: contribution `+0.009119`
- `lag_02__T2__flash_duration`: contribution `+0.004902`

### tick `68035`, seconds `31.50`, LSTM delta `-0.2169`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.027984`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `-0.023982`
- `lag_08__CT_place_LOWERTUNNEL`: contribution `-0.012085`
- `lag_02__CT_place_LOWERTUNNEL`: contribution `-0.008902`
- `lag_08__CT_place_CATWALK`: contribution `-0.006049`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.003379`

### tick `67331`, seconds `20.50`, LSTM delta `-0.2103`

Top all feature movements:
- `lag_00__T2__flash_duration`: contribution `-0.009119`
- `lag_10__CT_place_TOPOFMID`: contribution `-0.006858`
- `lag_06__T_flashes_last_5s`: contribution `-0.006763`
- `lag_02__T2__duck_amount`: contribution `-0.004756`
- `lag_14__CT1__flash_duration`: contribution `-0.004600`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `-0.009119`
- `lag_06__T_flashes_last_5s`: contribution `-0.006763`
- `lag_14__CT1__flash_duration`: contribution `-0.004600`

### tick `66915`, seconds `14.00`, LSTM delta `+0.1362`

Top all feature movements:
- `lag_06__CT_place_CATWALK`: contribution `+0.008465`
- `lag_00__CT4__is_scoped`: contribution `+0.005740`
- `lag_03__T_flashes_last_5s`: contribution `+0.005528`
- `lag_06__CT_place_MIDDLE`: contribution `+0.004581`
- `lag_00__kill_diff_last_3s`: contribution `+0.004210`

Top utility-only movements:
- `lag_03__T_flashes_last_5s`: contribution `+0.005528`
- `lag_12__CT1__flash_duration`: contribution `+0.003585`
- `lag_01__CT1__flash_duration`: contribution `+0.003061`
- `lag_12__T4__flash_duration`: contribution `+0.002699`

### tick `67011`, seconds `15.50`, LSTM delta `-0.0854`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.006763`
- `lag_08__CT_place_CATWALK`: contribution `-0.006049`
- `lag_14__T1__is_scoped`: contribution `-0.004604`
- `lag_00__kill_diff_last_3s`: contribution `-0.004210`
- `lag_00__T_kills_last_3s`: contribution `-0.004150`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.006763`
- `lag_04__CT1__flash_duration`: contribution `-0.003111`
