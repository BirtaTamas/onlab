# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `19`

## Largest probability jumps

- tick `145609`, seconds `46.50`, LSTM `0.3395`, delta `-0.2165`
- tick `144137`, seconds `23.50`, LSTM `0.3529`, delta `-0.1791`
- tick `145737`, seconds `48.50`, LSTM `0.0472`, delta `-0.1495`
- tick `144073`, seconds `22.50`, LSTM `0.5497`, delta `-0.1483`
- tick `144041`, seconds `22.00`, LSTM `0.6980`, delta `+0.0964`
- tick `144329`, seconds `26.50`, LSTM `0.4338`, delta `+0.0700`
- tick `144233`, seconds `25.00`, LSTM `0.2369`, delta `-0.0673`
- tick `145641`, seconds `47.00`, LSTM `0.2727`, delta `-0.0668`
- tick `144265`, seconds `25.50`, LSTM `0.3029`, delta `+0.0659`
- tick `144297`, seconds `26.00`, LSTM `0.3638`, delta `+0.0609`

## Top 15 local ridge features

- `lag_10__CT_place_MINI`: coefficient `-0.002152`, |coef| `0.002152`
- `lag_00__T_kills_last_3s`: coefficient `-0.002109`, |coef| `0.002109`
- `lag_11__T_place_SECRET`: coefficient `-0.001851`, |coef| `0.001851`
- `lag_03__CT2__flash_duration`: coefficient `0.001780`, |coef| `0.001780`
- `lag_00__kill_diff_last_3s`: coefficient `0.001719`, |coef| `0.001719`
- `lag_00__T_damage_last_5s`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_01__T5__duck_amount`: coefficient `-0.001602`, |coef| `0.001602`
- `lag_03__T_place_HUT`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_06__CT3__flash_duration`: coefficient `0.001524`, |coef| `0.001524`
- `lag_07__T_place_MINI`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_00__damage_diff_last_5s`: coefficient `0.001387`, |coef| `0.001387`
- `lag_14__CT_shots_fired_sum`: coefficient `0.001362`, |coef| `0.001362`
- `lag_06__CT_place_HEAVEN`: coefficient `-0.001347`, |coef| `0.001347`
- `lag_11__CT_place_LOCKERROOM`: coefficient `0.001332`, |coef| `0.001332`
- `lag_14__CT_place_MINI`: coefficient `-0.001317`, |coef| `0.001317`

## Top 10 utility ridge features

- `lag_03__CT2__flash_duration`: coefficient `0.001780` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001524` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001082` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000871` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000857` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000855` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000819` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000799` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000794` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.000790` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_MINI`: coefficient `-0.002152` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002109` (lowers CT win probability)
- `lag_11__T_place_SECRET`: coefficient `-0.001851` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001719` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001611` (lowers CT win probability)
- `lag_01__T5__duck_amount`: coefficient `-0.001602` (lowers CT win probability)
- `lag_03__T_place_HUT`: coefficient `-0.001564` (lowers CT win probability)
- `lag_07__T_place_MINI`: coefficient `-0.001490` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001387` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.001362` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `145609`, seconds `46.50`, LSTM delta `-0.2165`

Top all feature movements:
- `lag_03__CT2__flash_duration`: contribution `-0.014015`
- `lag_10__CT_place_MINI`: contribution `-0.013192`
- `lag_06__CT3__flash_duration`: contribution `-0.011746`
- `lag_11__T_place_SECRET`: contribution `-0.009740`
- `lag_06__CT_place_HEAVEN`: contribution `-0.007274`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.014015`
- `lag_06__CT3__flash_duration`: contribution `-0.011746`
- `lag_07__T5__flash_duration`: contribution `-0.005794`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002956`

### tick `144137`, seconds `23.50`, LSTM delta `-0.1791`

Top all feature movements:
- `lag_07__T_place_MINI`: contribution `-0.020734`
- `lag_13__CT_place_OBSERVATION`: contribution `-0.016957`
- `lag_11__CT_place_LOCKERROOM`: contribution `-0.016583`
- `lag_03__T_place_HUT`: contribution `-0.014581`
- `lag_03__T_place_MINI`: contribution `-0.010839`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `-0.002513`

### tick `145737`, seconds `48.50`, LSTM delta `-0.1495`

Top all feature movements:
- `lag_14__CT_place_MINI`: contribution `-0.008077`
- `lag_00__T_kills_last_3s`: contribution `-0.006682`
- `lag_10__CT3__flash_duration`: contribution `-0.006087`
- `lag_15__T_place_SECRET`: contribution `-0.005948`
- `lag_01__T5__duck_amount`: contribution `-0.005916`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.006087`
- `lag_07__CT2__flash_duration`: contribution `-0.004661`
- `lag_11__T5__flash_duration`: contribution `-0.003966`
- `lag_07__CT_flash_duration_sum`: contribution `-0.002312`

### tick `144073`, seconds `22.50`, LSTM delta `-0.1483`

Top all feature movements:
- `lag_11__CT_place_OBSERVATION`: contribution `-0.015075`
- `lag_03__T_place_HUT`: contribution `-0.014581`
- `lag_01__T_place_MINI`: contribution `-0.009636`
- `lag_15__CT_place_OBSERVATION`: contribution `-0.009511`
- `lag_12__CT_place_LOCKERROOM`: contribution `-0.009357`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.002564`

### tick `144041`, seconds `22.00`, LSTM delta `+0.0964`

Top all feature movements:
- `lag_11__CT_place_LOCKERROOM`: contribution `+0.016583`
- `lag_14__CT_place_OBSERVATION`: contribution `+0.009117`
- `lag_10__CT_place_OBSERVATION`: contribution `+0.008927`
- `lag_08__CT_place_LOCKERROOM`: contribution `+0.006301`
- `lag_04__T_place_MINI`: contribution `+0.005882`

Top utility-only movements:
- No utility movement among the top local contributors.
