# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `25156`, seconds `41.00`, LSTM `0.8089`, delta `+0.1197`
- tick `25252`, seconds `42.50`, LSTM `0.8313`, delta `+0.0995`
- tick `25220`, seconds `42.00`, LSTM `0.7318`, delta `-0.0977`
- tick `25860`, seconds `52.00`, LSTM `0.9236`, delta `+0.0515`
- tick `22884`, seconds `5.50`, LSTM `0.7495`, delta `+0.0404`
- tick `25540`, seconds `47.00`, LSTM `0.8329`, delta `+0.0387`
- tick `25828`, seconds `51.50`, LSTM `0.8720`, delta `+0.0376`
- tick `24804`, seconds `35.50`, LSTM `0.6804`, delta `-0.0348`
- tick `25732`, seconds `50.00`, LSTM `0.8516`, delta `+0.0267`
- tick `24932`, seconds `37.50`, LSTM `0.6917`, delta `+0.0259`

## Top 15 local ridge features

- `lag_02__T_place_ARCH`: coefficient `-0.001284`, |coef| `0.001284`
- `lag_14__T_flashed_players`: coefficient `0.001284`, |coef| `0.001284`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001124`, |coef| `0.001124`
- `lag_03__T_place_ARCH`: coefficient `0.000985`, |coef| `0.000985`
- `lag_00__kill_diff_last_3s`: coefficient `0.000972`, |coef| `0.000972`
- `lag_13__CT_place_BALCONY`: coefficient `-0.000922`, |coef| `0.000922`
- `lag_12__T_place_ARCH`: coefficient `-0.000871`, |coef| `0.000871`
- `lag_05__CT2__flash_duration`: coefficient `0.000851`, |coef| `0.000851`
- `lag_07__T_place_ARCH`: coefficient `0.000848`, |coef| `0.000848`
- `lag_07__CT2__flash_duration`: coefficient `-0.000800`, |coef| `0.000800`
- `lag_00__CT_kills_last_3s`: coefficient `0.000783`, |coef| `0.000783`
- `lag_00__damage_diff_last_5s`: coefficient `0.000782`, |coef| `0.000782`
- `lag_00__CT_place_QUAD`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_05__CT_place_RUINS`: coefficient `0.000772`, |coef| `0.000772`
- `lag_15__CT_place_RUINS`: coefficient `-0.000752`, |coef| `0.000752`

## Top 10 utility ridge features

- `lag_05__CT2__flash_duration`: coefficient `0.000851` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.000800` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000600` (lowers CT win probability)
- `lag_09__T_flashes_last_5s`: coefficient `0.000567` (raises CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000480` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000468` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.000436` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000355` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.000331` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.000330` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_ARCH`: coefficient `-0.001284` (lowers CT win probability)
- `lag_14__T_flashed_players`: coefficient `0.001284` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.001124` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.000985` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000972` (raises CT win probability)
- `lag_13__CT_place_BALCONY`: coefficient `-0.000922` (lowers CT win probability)
- `lag_12__T_place_ARCH`: coefficient `-0.000871` (lowers CT win probability)
- `lag_07__T_place_ARCH`: coefficient `0.000848` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000783` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000782` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25156`, seconds `41.00`, LSTM delta `+0.1197`

Top all feature movements:
- `lag_14__T_flashed_players`: contribution `+0.012389`
- `lag_02__T_place_ARCH`: contribution `+0.011950`
- `lag_07__T_place_ARCH`: contribution `+0.007886`
- `lag_05__CT2__flash_duration`: contribution `+0.005909`
- `lag_08__CT_place_BALCONY`: contribution `+0.004492`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.005909`

### tick `25252`, seconds `42.50`, LSTM delta `+0.0995`

Top all feature movements:
- `lag_02__T_place_ARCH`: contribution `+0.011950`
- `lag_13__T_place_ARCH`: contribution `+0.005573`
- `lag_05__T_place_ARCH`: contribution `+0.004934`
- `lag_10__T_place_ARCH`: contribution `+0.004051`
- `lag_11__T_place_ARCH`: contribution `+0.003935`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.001444`

### tick `25220`, seconds `42.00`, LSTM delta `-0.0977`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `-0.009167`
- `lag_12__T_place_ARCH`: contribution `-0.008100`
- `lag_13__CT_place_BALCONY`: contribution `-0.005916`
- `lag_07__CT2__flash_duration`: contribution `-0.005556`
- `lag_10__T_place_ARCH`: contribution `+0.004051`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.005556`

### tick `25860`, seconds `52.00`, LSTM delta `+0.0515`

Top all feature movements:
- `lag_01__T_place_CTSPAWN`: contribution `+0.006241`
- `lag_06__CT_place_QUAD`: contribution `+0.003738`
- `lag_08__CT_place_QUAD`: contribution `+0.003510`
- `lag_02__T_flashed_players`: contribution `+0.003117`
- `lag_14__CT2__flash_duration`: contribution `+0.003025`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `+0.003025`
- `lag_00__T1__flash_duration`: contribution `+0.001099`

### tick `22884`, seconds `5.50`, LSTM delta `+0.0404`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.005440`
- `lag_09__T_flashes_last_5s`: contribution `+0.005137`
- `lag_10__T_flashes_last_5s`: contribution `+0.004346`
- `lag_00__CT_place_LIBRARY`: contribution `+0.002111`
- `lag_01__T_place_LOWERMID`: contribution `+0.002086`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.005440`
- `lag_09__T_flashes_last_5s`: contribution `+0.005137`
- `lag_10__T_flashes_last_5s`: contribution `+0.004346`
- `lag_11__CT5__smoke`: contribution `-0.000427`
- `lag_11__CT_molly_inv`: contribution `+0.000350`
