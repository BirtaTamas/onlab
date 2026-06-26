# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `14`

## Largest probability jumps

- tick `116951`, seconds `61.00`, LSTM `0.8437`, delta `+0.1991`
- tick `116279`, seconds `50.50`, LSTM `0.6103`, delta `+0.0985`
- tick `116919`, seconds `60.50`, LSTM `0.6446`, delta `+0.0622`
- tick `117079`, seconds `63.00`, LSTM `0.9422`, delta `+0.0617`
- tick `116343`, seconds `51.50`, LSTM `0.6417`, delta `+0.0495`
- tick `116471`, seconds `53.50`, LSTM `0.6113`, delta `-0.0440`
- tick `114263`, seconds `19.00`, LSTM `0.6051`, delta `+0.0435`
- tick `114423`, seconds `21.50`, LSTM `0.5682`, delta `-0.0355`
- tick `114519`, seconds `23.00`, LSTM `0.5943`, delta `+0.0333`
- tick `116407`, seconds `52.50`, LSTM `0.6615`, delta `+0.0297`

## Top 15 local ridge features

- `lag_14__T_place_CONTROL`: coefficient `-0.002098`, |coef| `0.002098`
- `lag_15__T_place_CONTROL`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_07__CT_place_VENTS`: coefficient `0.001762`, |coef| `0.001762`
- `lag_02__CT_place_VENTS`: coefficient `-0.001706`, |coef| `0.001706`
- `lag_14__T_place_RAMP`: coefficient `0.001335`, |coef| `0.001335`
- `lag_00__T_place_RAMP`: coefficient `-0.001317`, |coef| `0.001317`
- `lag_07__CT_place_HEAVEN`: coefficient `0.001191`, |coef| `0.001191`
- `lag_15__T_place_RAMP`: coefficient `0.001137`, |coef| `0.001137`
- `lag_00__CT_kills_last_3s`: coefficient `0.001107`, |coef| `0.001107`
- `lag_07__CT_place_CATWALK`: coefficient `-0.001004`, |coef| `0.001004`
- `lag_14__CT_place_CATWALK`: coefficient `0.000979`, |coef| `0.000979`
- `lag_00__T4__has_bomb`: coefficient `-0.000947`, |coef| `0.000947`
- `lag_06__CT_place_OBSERVATION`: coefficient `0.000944`, |coef| `0.000944`
- `lag_00__kill_diff_last_3s`: coefficient `0.000923`, |coef| `0.000923`
- `lag_01__CT_place_DECON`: coefficient `0.000905`, |coef| `0.000905`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `-0.000681` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000650` (lowers CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `-0.000582` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000496` (lowers CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.000459` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000413` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000396` (lowers CT win probability)
- `lag_04__active_infernos_total`: coefficient `-0.000393` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000370` (lowers CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `-0.000324` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_CONTROL`: coefficient `-0.002098` (lowers CT win probability)
- `lag_15__T_place_CONTROL`: coefficient `-0.001812` (lowers CT win probability)
- `lag_07__CT_place_VENTS`: coefficient `0.001762` (raises CT win probability)
- `lag_02__CT_place_VENTS`: coefficient `-0.001706` (lowers CT win probability)
- `lag_14__T_place_RAMP`: coefficient `0.001335` (raises CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.001317` (lowers CT win probability)
- `lag_07__CT_place_HEAVEN`: coefficient `0.001191` (raises CT win probability)
- `lag_15__T_place_RAMP`: coefficient `0.001137` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001107` (raises CT win probability)
- `lag_07__CT_place_CATWALK`: coefficient `-0.001004` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `116951`, seconds `61.00`, LSTM delta `+0.1991`

Top all feature movements:
- `lag_14__T_place_CONTROL`: contribution `+0.014909`
- `lag_07__CT_place_VENTS`: contribution `+0.014786`
- `lag_02__CT_place_VENTS`: contribution `+0.014314`
- `lag_15__T_place_CONTROL`: contribution `+0.012873`
- `lag_07__CT_place_HEAVEN`: contribution `+0.006431`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.002026`

### tick `116279`, seconds `50.50`, LSTM delta `+0.0985`

Top all feature movements:
- `lag_00__T_place_RAMP`: contribution `+0.004658`
- `lag_11__T_place_TROPHY`: contribution `+0.004408`
- `lag_06__T_place_CONTROL`: contribution `+0.004328`
- `lag_01__T_place_CONTROL`: contribution `+0.004117`
- `lag_08__T_place_CONTROL`: contribution `+0.003847`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116919`, seconds `60.50`, LSTM delta `+0.0622`

Top all feature movements:
- `lag_14__T_place_CONTROL`: contribution `+0.014909`
- `lag_01__CT_place_VENTS`: contribution `+0.007338`
- `lag_06__CT_place_VENTS`: contribution `+0.004836`
- `lag_14__T_place_RAMP`: contribution `+0.004723`
- `lag_13__T_place_CONTROL`: contribution `+0.003652`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.001230`
- `lag_03__T_B_site_active_infernos`: contribution `+0.001121`

### tick `117079`, seconds `63.00`, LSTM delta `+0.0617`

Top all feature movements:
- `lag_01__CT_place_DECON`: contribution `+0.014384`
- `lag_00__T_place_RAMP`: contribution `+0.009316`
- `lag_06__CT_place_VENTS`: contribution `-0.004836`
- `lag_02__CT_place_RAFTERS`: contribution `-0.003696`
- `lag_00__CT_kills_last_3s`: contribution `+0.003196`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116343`, seconds `51.50`, LSTM delta `+0.0495`

Top all feature movements:
- `lag_14__CT_place_CRANE`: contribution `+0.008695`
- `lag_10__T_place_TROPHY`: contribution `+0.005322`
- `lag_08__T_place_CONTROL`: contribution `+0.003847`
- `lag_13__T_place_CONTROL`: contribution `-0.003652`
- `lag_08__T_place_TROPHY`: contribution `+0.003330`

Top utility-only movements:
- No utility movement among the top local contributors.
