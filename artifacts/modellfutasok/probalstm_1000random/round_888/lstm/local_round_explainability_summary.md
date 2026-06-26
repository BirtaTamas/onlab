# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `30`

## Largest probability jumps

- tick `238546`, seconds `54.50`, LSTM `0.8406`, delta `+0.1739`
- tick `236338`, seconds `20.00`, LSTM `0.7644`, delta `+0.1656`
- tick `235762`, seconds `11.00`, LSTM `0.6917`, delta `+0.1151`
- tick `238674`, seconds `56.50`, LSTM `0.9516`, delta `+0.0886`
- tick `236498`, seconds `22.50`, LSTM `0.7430`, delta `-0.0492`
- tick `236530`, seconds `23.00`, LSTM `0.7020`, delta `-0.0409`
- tick `237234`, seconds `34.00`, LSTM `0.6933`, delta `+0.0379`
- tick `237202`, seconds `33.50`, LSTM `0.6554`, delta `-0.0326`
- tick `237266`, seconds `34.50`, LSTM `0.6611`, delta `-0.0321`
- tick `236018`, seconds `15.00`, LSTM `0.6773`, delta `-0.0319`

## Top 15 local ridge features

- `lag_09__T_flashes_last_5s`: coefficient `-0.002637`, |coef| `0.002637`
- `lag_00__CT_kills_last_3s`: coefficient `0.002440`, |coef| `0.002440`
- `lag_00__kill_diff_last_3s`: coefficient `0.002397`, |coef| `0.002397`
- `lag_15__T_place_SIDEENTRANCE`: coefficient `0.002054`, |coef| `0.002054`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001942`, |coef| `0.001942`
- `lag_10__T_he_last_5s`: coefficient `-0.001765`, |coef| `0.001765`
- `lag_00__CT_damage_last_5s`: coefficient `0.001675`, |coef| `0.001675`
- `lag_01__CT1__flash_duration`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_00__T1__has_bomb`: coefficient `-0.001301`, |coef| `0.001301`
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.001281`, |coef| `0.001281`
- `lag_13__T_flashes_last_5s`: coefficient `-0.001262`, |coef| `0.001262`
- `lag_00__damage_diff_last_5s`: coefficient `0.001224`, |coef| `0.001224`
- `lag_05__CT_place_HOUSE`: coefficient `0.001198`, |coef| `0.001198`
- `lag_13__T_place_TSIDELOWER`: coefficient `-0.001180`, |coef| `0.001180`
- `lag_07__CT1__flash_duration`: coefficient `0.001179`, |coef| `0.001179`

## Top 10 utility ridge features

- `lag_09__T_flashes_last_5s`: coefficient `-0.002637` (lowers CT win probability)
- `lag_10__T_he_last_5s`: coefficient `-0.001765` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.001401` (lowers CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.001262` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.001179` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.001084` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.001026` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.001023` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000960` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.000906` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002440` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002397` (raises CT win probability)
- `lag_15__T_place_SIDEENTRANCE`: coefficient `0.002054` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001942` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001675` (raises CT win probability)
- `lag_00__T1__has_bomb`: coefficient `-0.001301` (lowers CT win probability)
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.001281` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001224` (raises CT win probability)
- `lag_05__CT_place_HOUSE`: coefficient `0.001198` (raises CT win probability)
- `lag_13__T_place_TSIDELOWER`: coefficient `-0.001180` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `238546`, seconds `54.50`, LSTM delta `+0.1739`

Top all feature movements:
- `lag_09__T_flashes_last_5s`: contribution `+0.023894`
- `lag_15__T_place_SIDEENTRANCE`: contribution `+0.010024`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.009480`
- `lag_00__CT_kills_last_3s`: contribution `+0.007043`
- `lag_00__kill_diff_last_3s`: contribution `+0.005770`

Top utility-only movements:
- `lag_09__T_flashes_last_5s`: contribution `+0.023894`
- `lag_01__CT1__flash_duration`: contribution `+0.004570`
- `lag_07__CT1__flash_duration`: contribution `+0.003845`
- `lag_09__T_B_site_active_infernos`: contribution `+0.003064`

### tick `236338`, seconds `20.00`, LSTM delta `+0.1656`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `+0.023037`
- `lag_00__CT_kills_last_3s`: contribution `+0.007043`
- `lag_10__T4__flash_duration`: contribution `+0.006306`
- `lag_00__kill_diff_last_3s`: contribution `+0.005770`
- `lag_06__CT4__flash_duration`: contribution `+0.005643`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `+0.023037`
- `lag_10__T4__flash_duration`: contribution `+0.006306`
- `lag_06__CT4__flash_duration`: contribution `+0.005643`
- `lag_13__CT2__flash_duration`: contribution `+0.004947`

### tick `235762`, seconds `11.00`, LSTM delta `+0.1151`

Top all feature movements:
- `lag_02__T_he_last_5s`: contribution `+0.009143`
- `lag_00__CT_kills_last_3s`: contribution `+0.007043`
- `lag_09__T_he_last_5s`: contribution `+0.006260`
- `lag_00__kill_diff_last_3s`: contribution `+0.005770`
- `lag_12__CT_place_HOUSE`: contribution `+0.004750`

Top utility-only movements:
- `lag_02__T_he_last_5s`: contribution `+0.009143`
- `lag_09__T_he_last_5s`: contribution `+0.006260`
- `lag_05__T1__flash_duration`: contribution `+0.002555`
- `lag_05__CT2__flash_duration`: contribution `+0.002369`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001289`

### tick `238674`, seconds `56.50`, LSTM delta `+0.0886`

Top all feature movements:
- `lag_13__T_flashes_last_5s`: contribution `+0.011436`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.009480`
- `lag_00__CT_kills_last_3s`: contribution `+0.007043`
- `lag_00__kill_diff_last_3s`: contribution `+0.005770`
- `lag_02__CT_place_ALLEY`: contribution `+0.002744`

Top utility-only movements:
- `lag_13__T_flashes_last_5s`: contribution `+0.011436`
- `lag_13__T_B_site_active_infernos`: contribution `+0.001618`
- `lag_11__CT1__flash_duration`: contribution `+0.001533`

### tick `236498`, seconds `22.50`, LSTM delta `-0.0492`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `-0.010145`
- `lag_00__kill_diff_last_3s`: contribution `-0.005770`
- `lag_15__T4__flash_duration`: contribution `-0.002764`
- `lag_11__T1__is_walking`: contribution `+0.002093`
- `lag_09__CT1__is_walking`: contribution `-0.002088`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `-0.010145`
- `lag_15__T4__flash_duration`: contribution `-0.002764`
- `lag_01__CT4__flash_duration`: contribution `-0.001959`
- `lag_11__CT4__flash_duration`: contribution `-0.001849`
