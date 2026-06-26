# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `33955`, seconds `86.00`, LSTM `0.3095`, delta `-0.2413`
- tick `32483`, seconds `63.00`, LSTM `0.1103`, delta `-0.2125`
- tick `33507`, seconds `79.00`, LSTM `0.5768`, delta `+0.2119`
- tick `34723`, seconds `98.00`, LSTM `0.0424`, delta `-0.1658`
- tick `32611`, seconds `65.00`, LSTM `0.2514`, delta `+0.1224`
- tick `32675`, seconds `66.00`, LSTM `0.4049`, delta `+0.1031`
- tick `33987`, seconds `86.50`, LSTM `0.2240`, delta `-0.0854`
- tick `34595`, seconds `96.00`, LSTM `0.1986`, delta `-0.0613`
- tick `32579`, seconds `64.50`, LSTM `0.1289`, delta `+0.0569`
- tick `31683`, seconds `50.50`, LSTM `0.2887`, delta `-0.0512`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004836`, |coef| `0.004836`
- `lag_00__T_kills_last_3s`: coefficient `-0.004298`, |coef| `0.004298`
- `lag_00__damage_diff_last_5s`: coefficient `0.004153`, |coef| `0.004153`
- `lag_05__T_place_LOWERTUNNEL`: coefficient `0.003982`, |coef| `0.003982`
- `lag_07__T_A_site_active_infernos`: coefficient `0.003857`, |coef| `0.003857`
- `lag_05__CT4__duck_amount`: coefficient `-0.003023`, |coef| `0.003023`
- `lag_08__T5__is_scoped`: coefficient `0.002564`, |coef| `0.002564`
- `lag_05__T_place_CATWALK`: coefficient `0.002415`, |coef| `0.002415`
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.002407`, |coef| `0.002407`
- `lag_03__CT_place_ARAMP`: coefficient `0.002392`, |coef| `0.002392`
- `lag_05__T_place_SHORTSTAIRS`: coefficient `-0.002385`, |coef| `0.002385`
- `lag_02__CT_place_MIDDOORS`: coefficient `-0.002365`, |coef| `0.002365`
- `lag_01__T_place_TUNNELSTAIRS`: coefficient `0.002254`, |coef| `0.002254`
- `lag_00__CT4__duck_amount`: coefficient `0.002183`, |coef| `0.002183`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.002182`, |coef| `0.002182`

## Top 10 utility ridge features

- `lag_07__T_A_site_active_infernos`: coefficient `0.003857` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.002407` (lowers CT win probability)
- `lag_07__T_active_infernos`: coefficient `0.002151` (raises CT win probability)
- `lag_10__T1__molly`: coefficient `-0.001567` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.001454` (raises CT win probability)
- `lag_07__active_infernos_total`: coefficient `0.001419` (raises CT win probability)
- `lag_09__CT5__smoke`: coefficient `-0.001337` (lowers CT win probability)
- `lag_13__CT_active_infernos`: coefficient `-0.001203` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `0.001117` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.001093` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004836` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004298` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004153` (raises CT win probability)
- `lag_05__T_place_LOWERTUNNEL`: coefficient `0.003982` (raises CT win probability)
- `lag_05__CT4__duck_amount`: coefficient `-0.003023` (lowers CT win probability)
- `lag_08__T5__is_scoped`: coefficient `0.002564` (raises CT win probability)
- `lag_05__T_place_CATWALK`: coefficient `0.002415` (raises CT win probability)
- `lag_03__CT_place_ARAMP`: coefficient `0.002392` (raises CT win probability)
- `lag_05__T_place_SHORTSTAIRS`: coefficient `-0.002385` (lowers CT win probability)
- `lag_02__CT_place_MIDDOORS`: coefficient `-0.002365` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `33955`, seconds `86.00`, LSTM delta `-0.2413`

Top all feature movements:
- `lag_05__T_place_LOWERTUNNEL`: contribution `-0.017216`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `-0.015737`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.015234`
- `lag_05__CT_place_OUTSIDELONG`: contribution `-0.014973`
- `lag_00__T_kills_last_3s`: contribution `-0.013615`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.011480`
- `lag_07__T_active_infernos`: contribution `-0.004481`

### tick `32483`, seconds `63.00`, LSTM delta `-0.2125`

Top all feature movements:
- `lag_03__CT_place_ARAMP`: contribution `-0.014903`
- `lag_00__T_kills_last_3s`: contribution `-0.013615`
- `lag_00__kill_diff_last_3s`: contribution `-0.011640`
- `lag_00__damage_diff_last_5s`: contribution `-0.009370`
- `lag_11__CT_flashed_players`: contribution `-0.008857`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33507`, seconds `79.00`, LSTM delta `+0.2119`

Top all feature movements:
- `lag_05__T_place_LOWERTUNNEL`: contribution `+0.017216`
- `lag_00__kill_diff_last_3s`: contribution `+0.011640`
- `lag_07__T_A_site_active_infernos`: contribution `+0.011480`
- `lag_05__T_place_SHORTSTAIRS`: contribution `+0.010023`
- `lag_00__damage_diff_last_5s`: contribution `+0.009370`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `+0.011480`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.008493`
- `lag_07__T_active_infernos`: contribution `+0.004481`
- `lag_10__T1__molly`: contribution `+0.003469`

### tick `34723`, seconds `98.00`, LSTM delta `-0.1658`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013615`
- `lag_08__T5__is_scoped`: contribution `-0.012228`
- `lag_00__kill_diff_last_3s`: contribution `-0.011640`
- `lag_05__CT4__duck_amount`: contribution `-0.011101`
- `lag_00__damage_diff_last_5s`: contribution `-0.009370`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32611`, seconds `65.00`, LSTM delta `+0.1224`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011640`
- `lag_07__CT_place_ARAMP`: contribution `+0.011330`
- `lag_05__CT4__duck_amount`: contribution `+0.008161`
- `lag_00__CT4__duck_amount`: contribution `-0.008016`
- `lag_01__T5__is_scoped`: contribution `+0.007645`

Top utility-only movements:
- No utility movement among the top local contributors.
