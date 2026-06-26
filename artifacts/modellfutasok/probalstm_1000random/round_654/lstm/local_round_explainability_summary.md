# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `30679`, seconds `21.00`, LSTM `0.4522`, delta `-0.2268`
- tick `30647`, seconds `20.50`, LSTM `0.6790`, delta `-0.1150`
- tick `30711`, seconds `21.50`, LSTM `0.3395`, delta `-0.1128`
- tick `30743`, seconds `22.00`, LSTM `0.2365`, delta `-0.1029`
- tick `31031`, seconds `26.50`, LSTM `0.0659`, delta `-0.0667`
- tick `29719`, seconds `6.00`, LSTM `0.8378`, delta `-0.0510`
- tick `30903`, seconds `24.50`, LSTM `0.1330`, delta `-0.0380`
- tick `30871`, seconds `24.00`, LSTM `0.1711`, delta `-0.0342`
- tick `30775`, seconds `22.50`, LSTM `0.2082`, delta `-0.0283`
- tick `29463`, seconds `2.00`, LSTM `0.8920`, delta `+0.0269`

## Top 15 local ridge features

- `lag_01__CT_place_BACKOFB`: coefficient `0.002574`, |coef| `0.002574`
- `lag_00__CT_place_BACKOFB`: coefficient `0.002507`, |coef| `0.002507`
- `lag_03__CT_place_LONGDOG`: coefficient `-0.002315`, |coef| `0.002315`
- `lag_08__T_place_BACKOFB`: coefficient `-0.002120`, |coef| `0.002120`
- `lag_09__T_place_BACKOFB`: coefficient `-0.001881`, |coef| `0.001881`
- `lag_07__T_place_BACKOFB`: coefficient `-0.001873`, |coef| `0.001873`
- `lag_00__T_kills_last_3s`: coefficient `-0.001839`, |coef| `0.001839`
- `lag_10__T_place_BACKOFB`: coefficient `-0.001820`, |coef| `0.001820`
- `lag_02__CT_place_BACKOFB`: coefficient `0.001711`, |coef| `0.001711`
- `lag_00__kill_diff_last_3s`: coefficient `0.001428`, |coef| `0.001428`
- `lag_00__T_damage_last_5s`: coefficient `-0.001372`, |coef| `0.001372`
- `lag_01__T_kills_last_3s`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_04__CT_A_site_active_infernos`: coefficient `0.001343`, |coef| `0.001343`
- `lag_06__T_place_BACKOFB`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_00__CT3__alive`: coefficient `0.001310`, |coef| `0.001310`

## Top 10 utility ridge features

- `lag_04__CT_A_site_active_infernos`: coefficient `0.001343` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.001221` (raises CT win probability)
- `lag_01__CT5__utility_total`: coefficient `0.001211` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001181` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.001034` (raises CT win probability)
- `lag_01__CT5__molly`: coefficient `0.001024` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000998` (raises CT win probability)
- `lag_01__CT5__smoke`: coefficient `0.000937` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000928` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000914` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_BACKOFB`: coefficient `0.002574` (raises CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.002507` (raises CT win probability)
- `lag_03__CT_place_LONGDOG`: coefficient `-0.002315` (lowers CT win probability)
- `lag_08__T_place_BACKOFB`: coefficient `-0.002120` (lowers CT win probability)
- `lag_09__T_place_BACKOFB`: coefficient `-0.001881` (lowers CT win probability)
- `lag_07__T_place_BACKOFB`: coefficient `-0.001873` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001839` (lowers CT win probability)
- `lag_10__T_place_BACKOFB`: coefficient `-0.001820` (lowers CT win probability)
- `lag_02__CT_place_BACKOFB`: coefficient `0.001711` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001428` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `30679`, seconds `21.00`, LSTM delta `-0.2268`

Top all feature movements:
- `lag_03__CT_place_LONGDOG`: contribution `-0.015102`
- `lag_01__CT_place_BACKOFB`: contribution `-0.014698`
- `lag_00__CT_place_BACKOFB`: contribution `-0.014312`
- `lag_11__CT_place_ELECTRICALBOX`: contribution `-0.009502`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `-0.006180`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `-0.004739`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.004194`
- `lag_01__CT5__utility_total`: contribution `-0.003432`

### tick `30647`, seconds `20.50`, LSTM delta `-0.1150`

Top all feature movements:
- `lag_00__CT_place_BACKOFB`: contribution `-0.014312`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.008261`
- `lag_02__CT_place_LONGDOG`: contribution `-0.007782`
- `lag_15__CT_place_ELECTRICALBOX`: contribution `+0.006180`
- `lag_00__T_kills_last_3s`: contribution `-0.005825`

Top utility-only movements:
- `lag_00__CT5__utility_total`: contribution `-0.003347`
- `lag_00__CT5__molly`: contribution `-0.002564`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.002401`

### tick `30711`, seconds `21.50`, LSTM delta `-0.1128`

Top all feature movements:
- `lag_01__CT_place_BACKOFB`: contribution `-0.014698`
- `lag_02__CT_place_BACKOFB`: contribution `-0.009770`
- `lag_11__CT_place_ELECTRICALBOX`: contribution `+0.009502`
- `lag_04__CT_place_LONGDOG`: contribution `-0.007004`
- `lag_08__T_place_BACKOFB`: contribution `-0.005694`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30743`, seconds `22.00`, LSTM delta `-0.1029`

Top all feature movements:
- `lag_02__CT_place_BACKOFB`: contribution `-0.009770`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.006078`
- `lag_09__T_place_BACKOFB`: contribution `-0.005051`
- `lag_07__T_place_BACKOFB`: contribution `-0.005031`
- `lag_10__T_place_BACKOFB`: contribution `-0.004888`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `-0.001848`

### tick `31031`, seconds `26.50`, LSTM delta `-0.0667`

Top all feature movements:
- `lag_06__T2__is_scoped`: contribution `-0.007230`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `-0.005957`
- `lag_01__T_place_LONGDOG`: contribution `-0.005641`
- `lag_14__CT_place_LONGDOG`: contribution `-0.003934`
- `lag_13__T_place_BACKOFB`: contribution `-0.003088`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `+0.002931`
- `lag_01__T_A_site_active_infernos`: contribution `-0.001668`
