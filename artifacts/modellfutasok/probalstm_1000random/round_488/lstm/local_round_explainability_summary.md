# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `15426`, seconds `34.50`, LSTM `0.8662`, delta `+0.2195`
- tick `15330`, seconds `33.00`, LSTM `0.6326`, delta `+0.2082`
- tick `16258`, seconds `47.50`, LSTM `0.8783`, delta `+0.1666`
- tick `16066`, seconds `44.50`, LSTM `0.7207`, delta `-0.0513`
- tick `15746`, seconds `39.50`, LSTM `0.7943`, delta `-0.0502`
- tick `14434`, seconds `19.00`, LSTM `0.3730`, delta `-0.0477`
- tick `16002`, seconds `43.50`, LSTM `0.7666`, delta `-0.0473`
- tick `17570`, seconds `68.00`, LSTM `0.9682`, delta `+0.0380`
- tick `14402`, seconds `18.50`, LSTM `0.4206`, delta `-0.0365`
- tick `14274`, seconds `16.50`, LSTM `0.4675`, delta `+0.0298`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003443`, |coef| `0.003443`
- `lag_00__CT_kills_last_3s`: coefficient `0.002995`, |coef| `0.002995`
- `lag_00__kill_diff_last_3s`: coefficient `0.002665`, |coef| `0.002665`
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002099`, |coef| `0.002099`
- `lag_00__CT4__shots_fired`: coefficient `0.002057`, |coef| `0.002057`
- `lag_01__T1__shots_fired`: coefficient `0.002015`, |coef| `0.002015`
- `lag_00__T4__flash`: coefficient `-0.001971`, |coef| `0.001971`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001924`, |coef| `0.001924`
- `lag_01__T_shots_fired_sum`: coefficient `0.001869`, |coef| `0.001869`
- `lag_00__CT_damage_last_5s`: coefficient `0.001866`, |coef| `0.001866`
- `lag_00__T1__shots_fired`: coefficient `0.001856`, |coef| `0.001856`
- `lag_00__T4__alive`: coefficient `-0.001783`, |coef| `0.001783`
- `lag_02__T2__is_walking`: coefficient `-0.001759`, |coef| `0.001759`
- `lag_00__T4__utility_total`: coefficient `-0.001754`, |coef| `0.001754`
- `lag_00__T4__hp`: coefficient `-0.001749`, |coef| `0.001749`

## Top 10 utility ridge features

- `lag_00__T4__flash`: coefficient `-0.001971` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001754` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.001584` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001577` (lowers CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.001153` (lowers CT win probability)
- `lag_11__T2__flash`: coefficient `-0.001000` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000958` (raises CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000931` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000895` (lowers CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000887` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003443` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002995` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002665` (raises CT win probability)
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002099` (lowers CT win probability)
- `lag_00__CT4__shots_fired`: coefficient `0.002057` (raises CT win probability)
- `lag_01__T1__shots_fired`: coefficient `0.002015` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001924` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `0.001869` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001866` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.001856` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `15426`, seconds `34.50`, LSTM delta `+0.2195`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.023922`
- `lag_00__T_shots_fired_sum`: contribution `+0.011538`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010025`
- `lag_01__T_shots_fired_sum`: contribution `+0.009807`
- `lag_00__CT_kills_last_3s`: contribution `+0.008647`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15330`, seconds `33.00`, LSTM delta `+0.2082`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.019138`
- `lag_00__CT_kills_last_3s`: contribution `+0.008647`
- `lag_00__kill_diff_last_3s`: contribution `+0.006414`
- `lag_09__CT_place_RUINS`: contribution `+0.005563`
- `lag_00__CT4__shots_fired`: contribution `+0.005540`

Top utility-only movements:
- `lag_00__T4__flash`: contribution `+0.005356`
- `lag_07__T_B_site_active_infernos`: contribution `+0.004478`
- `lag_00__T4__utility_total`: contribution `+0.004091`

### tick `16258`, seconds `47.50`, LSTM delta `+0.1666`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.014353`
- `lag_00__T_shots_fired_sum`: contribution `+0.011538`
- `lag_01__T_shots_fired_sum`: contribution `+0.009807`
- `lag_00__CT_kills_last_3s`: contribution `+0.008647`
- `lag_00__kill_diff_last_3s`: contribution `+0.006414`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.005385`
- `lag_08__CT5__flash_duration`: contribution `+0.005190`
- `lag_08__CT1__flash_duration`: contribution `+0.004793`
- `lag_08__CT_flash_duration_sum`: contribution `+0.004103`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.003290`

### tick `16066`, seconds `44.50`, LSTM delta `-0.0513`

Top all feature movements:
- `lag_01__CT2__is_walking`: contribution `-0.003873`
- `lag_02__CT1__flash_duration`: contribution `-0.003124`
- `lag_03__CT_place_ARCH`: contribution `-0.003083`
- `lag_05__T3__is_walking`: contribution `-0.002588`
- `lag_08__CT5__is_walking`: contribution `-0.002507`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.003124`
- `lag_02__CT_flash_duration_sum`: contribution `-0.001860`
- `lag_02__CT5__flash_duration`: contribution `-0.001296`

### tick `15746`, seconds `39.50`, LSTM delta `-0.0502`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `-0.008221`
- `lag_05__CT_place_BALCONY`: contribution `+0.005189`
- `lag_07__CT_place_BALCONY`: contribution `-0.004573`
- `lag_11__CT2__is_walking`: contribution `-0.003680`
- `lag_13__T4__duck_amount`: contribution `-0.003497`

Top utility-only movements:
- No utility movement among the top local contributors.
