# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `46879`, seconds `24.00`, LSTM `0.8121`, delta `+0.1312`
- tick `46975`, seconds `25.50`, LSTM `0.8719`, delta `+0.0728`
- tick `47039`, seconds `26.50`, LSTM `0.9299`, delta `+0.0661`
- tick `46207`, seconds `13.50`, LSTM `0.6784`, delta `-0.0614`
- tick `45919`, seconds `9.00`, LSTM `0.7144`, delta `+0.0281`
- tick `45375`, seconds `0.50`, LSTM `0.7528`, delta `+0.0264`
- tick `46271`, seconds `14.50`, LSTM `0.6529`, delta `-0.0250`
- tick `47199`, seconds `29.00`, LSTM `0.9482`, delta `+0.0225`
- tick `46367`, seconds `16.00`, LSTM `0.6776`, delta `+0.0222`
- tick `45951`, seconds `9.50`, LSTM `0.7362`, delta `+0.0219`

## Top 15 local ridge features

- `lag_00__CT_flashed_players`: coefficient `0.001171`, |coef| `0.001171`
- `lag_05__CT_place_MINI`: coefficient `0.000920`, |coef| `0.000920`
- `lag_13__T_place_SQUEAKY`: coefficient `0.000868`, |coef| `0.000868`
- `lag_12__CT_place_HELL`: coefficient `0.000767`, |coef| `0.000767`
- `lag_04__CT_place_HEAVEN`: coefficient `-0.000765`, |coef| `0.000765`
- `lag_00__CT_kills_last_3s`: coefficient `0.000750`, |coef| `0.000750`
- `lag_08__CT_place_HEAVEN`: coefficient `0.000719`, |coef| `0.000719`
- `lag_03__CT_flashed_players`: coefficient `0.000711`, |coef| `0.000711`
- `lag_00__T_place_LOBBY`: coefficient `-0.000698`, |coef| `0.000698`
- `lag_00__CT4__flash_duration`: coefficient `0.000696`, |coef| `0.000696`
- `lag_06__bomb_events_last_5s`: coefficient `-0.000692`, |coef| `0.000692`
- `lag_00__CT_place_MINI`: coefficient `0.000684`, |coef| `0.000684`
- `lag_04__CT_place_RAFTERS`: coefficient `0.000683`, |coef| `0.000683`
- `lag_00__T_A_site_active_infernos`: coefficient `0.000679`, |coef| `0.000679`
- `lag_02__T_place_SQUEAKY`: coefficient `-0.000677`, |coef| `0.000677`

## Top 10 utility ridge features

- `lag_00__CT4__flash_duration`: coefficient `0.000696` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.000679` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.000645` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000629` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000475` (raises CT win probability)
- `lag_04__CT2__molly`: coefficient `-0.000439` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000422` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000402` (lowers CT win probability)
- `lag_03__T5__molly`: coefficient `-0.000395` (lowers CT win probability)
- `lag_04__T4__molly`: coefficient `-0.000394` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_flashed_players`: coefficient `0.001171` (raises CT win probability)
- `lag_05__CT_place_MINI`: coefficient `0.000920` (raises CT win probability)
- `lag_13__T_place_SQUEAKY`: coefficient `0.000868` (raises CT win probability)
- `lag_12__CT_place_HELL`: coefficient `0.000767` (raises CT win probability)
- `lag_04__CT_place_HEAVEN`: coefficient `-0.000765` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000750` (raises CT win probability)
- `lag_08__CT_place_HEAVEN`: coefficient `0.000719` (raises CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.000711` (raises CT win probability)
- `lag_00__T_place_LOBBY`: coefficient `-0.000698` (lowers CT win probability)
- `lag_06__bomb_events_last_5s`: coefficient `-0.000692` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `46879`, seconds `24.00`, LSTM delta `+0.1312`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `+0.010260`
- `lag_05__CT_place_MINI`: contribution `+0.005640`
- `lag_13__T_place_SQUEAKY`: contribution `+0.005404`
- `lag_02__T_place_SQUEAKY`: contribution `+0.004216`
- `lag_00__CT_place_MINI`: contribution `+0.004195`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.004044`
- `lag_00__T_B_site_active_infernos`: contribution `+0.003649`
- `lag_00__CT4__flash_duration`: contribution `+0.003490`
- `lag_00__CT_flash_duration_sum`: contribution `+0.002552`
- `lag_00__T_active_infernos`: contribution `+0.001980`

### tick `46975`, seconds `25.50`, LSTM delta `+0.0728`

Top all feature movements:
- `lag_03__CT_flashed_players`: contribution `+0.006231`
- `lag_02__T_place_HUT`: contribution `+0.005635`
- `lag_03__CT_place_MINI`: contribution `+0.002657`
- `lag_00__CT_flashed_players`: contribution `-0.002565`
- `lag_08__CT_place_MINI`: contribution `+0.002349`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.002022`
- `lag_00__T_B_site_active_infernos`: contribution `-0.001824`
- `lag_03__CT4__flash_duration`: contribution `+0.001491`
- `lag_03__CT_flash_duration_sum`: contribution `+0.001327`
- `lag_03__T_A_site_active_infernos`: contribution `+0.001158`

### tick `47039`, seconds `26.50`, LSTM delta `+0.0661`

Top all feature movements:
- `lag_05__CT_place_MINI`: contribution `+0.005640`
- `lag_04__T_place_HUT`: contribution `+0.003518`
- `lag_05__CT_flashed_players`: contribution `+0.003113`
- `lag_01__T_place_HUT`: contribution `+0.003081`
- `lag_00__CT_flashed_players`: contribution `+0.002565`

Top utility-only movements:
- `lag_00__CT_flash_duration_sum`: contribution `+0.002084`
- `lag_00__CT3__flash_duration`: contribution `+0.001461`
- `lag_00__CT4__flash_duration`: contribution `-0.001076`

### tick `46207`, seconds `13.50`, LSTM delta `-0.0614`

Top all feature movements:
- `lag_11__CT_place_GARAGE`: contribution `-0.004009`
- `lag_14__CT_place_HELL`: contribution `-0.003742`
- `lag_10__T_place_ROOF`: contribution `-0.003346`
- `lag_15__T_flashes_last_5s`: contribution `-0.003256`
- `lag_14__CT_place_ADMIN`: contribution `-0.002620`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `-0.003256`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.002392`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.002271`
- `lag_08__CT_active_infernos`: contribution `-0.001101`

### tick `45919`, seconds `9.00`, LSTM delta `+0.0281`

Top all feature movements:
- `lag_05__CT_place_HELL`: contribution `+0.005211`
- `lag_04__CT_place_HEAVEN`: contribution `-0.004129`
- `lag_08__CT_place_HELL`: contribution `-0.003234`
- `lag_13__CT_place_OUTSIDE`: contribution `+0.002821`
- `lag_06__T_flashes_last_5s`: contribution `+0.002319`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.002319`
