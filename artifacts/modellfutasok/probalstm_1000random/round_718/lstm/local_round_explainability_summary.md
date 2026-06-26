# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `11`

## Largest probability jumps

- tick `87455`, seconds `37.00`, LSTM `0.8816`, delta `+0.2340`
- tick `86079`, seconds `15.50`, LSTM `0.7372`, delta `+0.1534`
- tick `87647`, seconds `40.00`, LSTM `0.9518`, delta `+0.1333`
- tick `86207`, seconds `17.50`, LSTM `0.7520`, delta `+0.1090`
- tick `86399`, seconds `20.50`, LSTM `0.6093`, delta `-0.0895`
- tick `87519`, seconds `38.00`, LSTM `0.8252`, delta `-0.0573`
- tick `86111`, seconds `16.00`, LSTM `0.6843`, delta `-0.0528`
- tick `86303`, seconds `19.00`, LSTM `0.7535`, delta `-0.0479`
- tick `87583`, seconds `39.00`, LSTM `0.8586`, delta `+0.0447`
- tick `86143`, seconds `16.50`, LSTM `0.6404`, delta `-0.0440`

## Top 15 local ridge features

- `lag_11__CT_place_OBSERVATION`: coefficient `0.002309`, |coef| `0.002309`
- `lag_04__CT_place_OBSERVATION`: coefficient `-0.002293`, |coef| `0.002293`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002013`, |coef| `0.002013`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001625`, |coef| `0.001625`
- `lag_00__CT1__shots_fired`: coefficient `0.001566`, |coef| `0.001566`
- `lag_00__damage_diff_last_5s`: coefficient `0.001500`, |coef| `0.001500`
- `lag_11__CT_place_TUNNELS`: coefficient `0.001490`, |coef| `0.001490`
- `lag_11__CT_place_VENTS`: coefficient `-0.001410`, |coef| `0.001410`
- `lag_00__CT_kills_last_3s`: coefficient `0.001399`, |coef| `0.001399`
- `lag_00__CT_damage_last_5s`: coefficient `0.001383`, |coef| `0.001383`
- `lag_00__kill_diff_last_3s`: coefficient `0.001371`, |coef| `0.001371`
- `lag_05__CT_place_LOBBY`: coefficient `0.001356`, |coef| `0.001356`
- `lag_05__CT_place_HUT`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_10__CT_place_OBSERVATION`: coefficient `-0.001222`, |coef| `0.001222`
- `lag_13__CT_place_ADMIN`: coefficient `-0.001156`, |coef| `0.001156`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `0.000780` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000746` (raises CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.000737` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000706` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.000615` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `-0.000578` (lowers CT win probability)
- `lag_04__T_active_smokes`: coefficient `-0.000574` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000557` (lowers CT win probability)
- `lag_05__T_A_site_active_smokes`: coefficient `-0.000511` (lowers CT win probability)
- `lag_11__CT_active_infernos`: coefficient `0.000495` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_OBSERVATION`: coefficient `0.002309` (raises CT win probability)
- `lag_04__CT_place_OBSERVATION`: coefficient `-0.002293` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002013` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001625` (raises CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `0.001566` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001500` (raises CT win probability)
- `lag_11__CT_place_TUNNELS`: coefficient `0.001490` (raises CT win probability)
- `lag_11__CT_place_VENTS`: coefficient `-0.001410` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001399` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001383` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `87455`, seconds `37.00`, LSTM delta `+0.2340`

Top all feature movements:
- `lag_11__CT_place_OBSERVATION`: contribution `+0.040213`
- `lag_04__CT_place_OBSERVATION`: contribution `+0.039927`
- `lag_11__CT_place_VENTS`: contribution `+0.011833`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009789`
- `lag_13__CT_place_ADMIN`: contribution `+0.008033`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86079`, seconds `15.50`, LSTM delta `+0.1534`

Top all feature movements:
- `lag_05__CT_place_HUT`: contribution `+0.011962`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011187`
- `lag_05__CT_place_LOBBY`: contribution `+0.011098`
- `lag_13__CT_place_ADMIN`: contribution `+0.008033`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007902`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `+0.002632`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.002492`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001967`

### tick `87647`, seconds `40.00`, LSTM delta `+0.1333`

Top all feature movements:
- `lag_10__CT_place_OBSERVATION`: contribution `+0.021274`
- `lag_05__CT_shots_fired_sum`: contribution `+0.006405`
- `lag_07__CT_shots_fired_sum`: contribution `+0.003882`
- `lag_10__CT_place_TUNNELS`: contribution `+0.003501`
- `lag_14__CT3__is_scoped`: contribution `+0.003439`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86207`, seconds `17.50`, LSTM delta `+0.1090`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `+0.011252`
- `lag_13__CT_place_HUT`: contribution `+0.009566`
- `lag_02__CT_place_HUT`: contribution `+0.007696`
- `lag_00__T_flashes_last_5s`: contribution `+0.007070`
- `lag_03__CT1__shots_fired`: contribution `+0.006713`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.007070`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.002632`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001967`

### tick `86399`, seconds `20.50`, LSTM delta `-0.0895`

Top all feature movements:
- `lag_15__CT_place_HUT`: contribution `-0.006498`
- `lag_08__CT_place_HUT`: contribution `-0.006220`
- `lag_09__CT_shots_fired_sum`: contribution `-0.005883`
- `lag_06__T_flashes_last_5s`: contribution `-0.005570`
- `lag_03__CT_place_HUT`: contribution `-0.005348`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.005570`
