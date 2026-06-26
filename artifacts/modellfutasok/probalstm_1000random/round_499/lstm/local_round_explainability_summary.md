# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `22011`, seconds `11.50`, LSTM `0.3669`, delta `-0.1595`
- tick `26811`, seconds `86.50`, LSTM `0.0213`, delta `-0.1014`
- tick `23131`, seconds `29.00`, LSTM `0.3583`, delta `+0.0895`
- tick `22043`, seconds `12.00`, LSTM `0.3121`, delta `-0.0548`
- tick `26555`, seconds `82.50`, LSTM `0.0997`, delta `-0.0448`
- tick `26683`, seconds `84.50`, LSTM `0.1596`, delta `+0.0442`
- tick `22171`, seconds `14.00`, LSTM `0.2938`, delta `-0.0435`
- tick `23163`, seconds `29.50`, LSTM `0.3151`, delta `-0.0433`
- tick `23035`, seconds `27.50`, LSTM `0.2813`, delta `-0.0367`
- tick `23675`, seconds `37.50`, LSTM `0.2850`, delta `-0.0317`

## Top 15 local ridge features

- `lag_08__T_place_UNDERA`: coefficient `-0.001924`, |coef| `0.001924`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001395`, |coef| `0.001395`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_02__CT2__shots_fired`: coefficient `-0.000981`, |coef| `0.000981`
- `lag_00__T_place_UNDERA`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_05__T_flashed_players`: coefficient `-0.000928`, |coef| `0.000928`
- `lag_07__T_place_UNDERA`: coefficient `-0.000905`, |coef| `0.000905`
- `lag_04__T4__flash_duration`: coefficient `-0.000862`, |coef| `0.000862`
- `lag_01__CT2__shots_fired`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_09__T_place_UNDERA`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.000823`, |coef| `0.000823`
- `lag_02__T_place_EXTENDEDA`: coefficient `-0.000784`, |coef| `0.000784`
- `lag_04__CT_place_UPPERTUNNEL`: coefficient `-0.000759`, |coef| `0.000759`
- `lag_03__CT2__shots_fired`: coefficient `-0.000757`, |coef| `0.000757`
- `lag_00__CT_velocity_mean`: coefficient `-0.000726`, |coef| `0.000726`

## Top 10 utility ridge features

- `lag_04__T4__flash_duration`: coefficient `-0.000862` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.000574` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `-0.000549` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000543` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000539` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.000532` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000500` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.000486` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.000482` (lowers CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000482` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_UNDERA`: coefficient `-0.001924` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001395` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.001063` (lowers CT win probability)
- `lag_02__CT2__shots_fired`: coefficient `-0.000981` (lowers CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.000971` (lowers CT win probability)
- `lag_05__T_flashed_players`: coefficient `-0.000928` (lowers CT win probability)
- `lag_07__T_place_UNDERA`: coefficient `-0.000905` (lowers CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.000848` (lowers CT win probability)
- `lag_09__T_place_UNDERA`: coefficient `-0.000848` (lowers CT win probability)
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.000823` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22011`, seconds `11.50`, LSTM delta `-0.1595`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012603`
- `lag_04__T4__flash_duration`: contribution `-0.005168`
- `lag_01__CT3__flash_duration`: contribution `-0.003858`
- `lag_06__CT_place_SHORTSTAIRS`: contribution `-0.003641`
- `lag_10__T_place_OUTSIDETUNNEL`: contribution `-0.003166`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `-0.005168`
- `lag_01__CT3__flash_duration`: contribution `-0.003858`
- `lag_04__T_flash_duration_sum`: contribution `-0.002421`
- `lag_04__T5__flash_duration`: contribution `-0.002418`
- `lag_06__CT5__flash_duration`: contribution `-0.002145`

### tick `26811`, seconds `86.50`, LSTM delta `-0.1014`

Top all feature movements:
- `lag_08__T_place_UNDERA`: contribution `-0.030067`
- `lag_05__T_flashed_players`: contribution `-0.007164`
- `lag_02__T_place_EXTENDEDA`: contribution `-0.003888`
- `lag_03__T_flashed_players`: contribution `-0.002790`
- `lag_05__CT5__flash_duration`: contribution `-0.002378`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.002378`
- `lag_05__T4__flash_duration`: contribution `-0.001432`
- `lag_00__T4__flash_duration`: contribution `-0.001405`
- `lag_08__T_A_site_active_infernos`: contribution `-0.001256`
- `lag_05__T_flash_duration_sum`: contribution `-0.001174`

### tick `23131`, seconds `29.00`, LSTM delta `+0.0895`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `+0.006623`
- `lag_04__CT_place_UPPERTUNNEL`: contribution `+0.005820`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005817`
- `lag_10__CT_place_UPPERTUNNEL`: contribution `+0.004758`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `+0.004166`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22043`, seconds `12.00`, LSTM delta `-0.0548`

Top all feature movements:
- `lag_05__T4__flash_duration`: contribution `-0.003293`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `-0.002910`
- `lag_02__CT2__shots_fired`: contribution `-0.002438`
- `lag_05__T5__flash_duration`: contribution `-0.002401`
- `lag_03__CT2__shots_fired`: contribution `-0.002259`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.003293`
- `lag_05__T5__flash_duration`: contribution `-0.002401`
- `lag_05__T_flash_duration_sum`: contribution `-0.002149`
- `lag_02__CT3__flash_duration`: contribution `-0.001238`

### tick `26555`, seconds `82.50`, LSTM delta `-0.0448`

Top all feature movements:
- `lag_00__T_place_UNDERA`: contribution `-0.015170`
- `lag_13__T_place_EXTENDEDA`: contribution `-0.002371`
- `lag_08__CT4__is_scoped`: contribution `-0.001941`
- `lag_10__T_place_SHORTSTAIRS`: contribution `+0.001630`
- `lag_10__T_place_EXTENDEDA`: contribution `-0.001609`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.000809`
