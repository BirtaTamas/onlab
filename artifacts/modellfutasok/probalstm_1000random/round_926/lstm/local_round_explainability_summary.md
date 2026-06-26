# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `24`

## Largest probability jumps

- tick `219187`, seconds `85.00`, LSTM `0.2689`, delta `-0.2832`
- tick `219123`, seconds `84.00`, LSTM `0.6437`, delta `+0.1927`
- tick `219283`, seconds `86.50`, LSTM `0.0392`, delta `-0.1766`
- tick `218707`, seconds `77.50`, LSTM `0.5253`, delta `-0.1131`
- tick `218643`, seconds `76.50`, LSTM `0.6356`, delta `+0.1092`
- tick `219155`, seconds `84.50`, LSTM `0.5521`, delta `-0.0916`
- tick `219219`, seconds `85.50`, LSTM `0.2095`, delta `-0.0594`
- tick `219091`, seconds `83.50`, LSTM `0.4510`, delta `+0.0557`
- tick `219027`, seconds `82.50`, LSTM `0.4254`, delta `-0.0481`
- tick `218099`, seconds `68.00`, LSTM `0.5034`, delta `+0.0388`

## Top 15 local ridge features

- `lag_03__T_place_HUT`: coefficient `0.002542`, |coef| `0.002542`
- `lag_07__T_place_SQUEAKY`: coefficient `-0.002215`, |coef| `0.002215`
- `lag_14__CT_place_VENTS`: coefficient `-0.002188`, |coef| `0.002188`
- `lag_00__kill_diff_last_3s`: coefficient `0.001942`, |coef| `0.001942`
- `lag_15__CT_place_RAFTERS`: coefficient `0.001893`, |coef| `0.001893`
- `lag_00__T_kills_last_3s`: coefficient `-0.001761`, |coef| `0.001761`
- `lag_02__T_place_MINI`: coefficient `0.001690`, |coef| `0.001690`
- `lag_14__T_place_MINI`: coefficient `-0.001521`, |coef| `0.001521`
- `lag_07__T_place_HUT`: coefficient `-0.001385`, |coef| `0.001385`
- `lag_06__T_place_SQUEAKY`: coefficient `-0.001333`, |coef| `0.001333`
- `lag_00__CT_place_VENTS`: coefficient `0.001268`, |coef| `0.001268`
- `lag_01__T_kills_last_3s`: coefficient `-0.001218`, |coef| `0.001218`
- `lag_15__damage_diff_last_5s`: coefficient `0.001185`, |coef| `0.001185`
- `lag_15__T_place_MINI`: coefficient `-0.001161`, |coef| `0.001161`
- `lag_12__CT_place_VENTS`: coefficient `0.001149`, |coef| `0.001149`

## Top 10 utility ridge features

- `lag_07__T_A_site_active_infernos`: coefficient `-0.000808` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.000764` (lowers CT win probability)
- `lag_09__T3__molly`: coefficient `0.000684` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `0.000549` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.000527` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.000457` (lowers CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000447` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000404` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000381` (lowers CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.000373` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_HUT`: coefficient `0.002542` (raises CT win probability)
- `lag_07__T_place_SQUEAKY`: coefficient `-0.002215` (lowers CT win probability)
- `lag_14__CT_place_VENTS`: coefficient `-0.002188` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001942` (raises CT win probability)
- `lag_15__CT_place_RAFTERS`: coefficient `0.001893` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001761` (lowers CT win probability)
- `lag_02__T_place_MINI`: coefficient `0.001690` (raises CT win probability)
- `lag_14__T_place_MINI`: coefficient `-0.001521` (lowers CT win probability)
- `lag_07__T_place_HUT`: coefficient `-0.001385` (lowers CT win probability)
- `lag_06__T_place_SQUEAKY`: coefficient `-0.001333` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `219187`, seconds `85.00`, LSTM delta `-0.2832`

Top all feature movements:
- `lag_03__T_place_HUT`: contribution `-0.023691`
- `lag_14__CT_place_VENTS`: contribution `-0.018357`
- `lag_07__T_place_SQUEAKY`: contribution `-0.013789`
- `lag_07__T_place_HUT`: contribution `-0.012909`
- `lag_15__CT_place_RAFTERS`: contribution `-0.010115`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `219123`, seconds `84.00`, LSTM delta `+0.1927`

Top all feature movements:
- `lag_03__T_place_HUT`: contribution `+0.023691`
- `lag_15__T_place_MINI`: contribution `+0.016154`
- `lag_01__T_place_HUT`: contribution `+0.009835`
- `lag_12__CT_place_VENTS`: contribution `+0.009640`
- `lag_00__kill_diff_last_3s`: contribution `+0.004675`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `219283`, seconds `86.50`, LSTM delta `-0.1766`

Top all feature movements:
- `lag_00__T_place_MINI`: contribution `-0.013080`
- `lag_00__CT_place_VENTS`: contribution `-0.010638`
- `lag_15__CT_place_RAFTERS`: contribution `-0.010115`
- `lag_08__T_place_HUT`: contribution `-0.009314`
- `lag_10__T_place_SQUEAKY`: contribution `-0.007038`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `218707`, seconds `77.50`, LSTM delta `-0.1131`

Top all feature movements:
- `lag_02__T_place_MINI`: contribution `-0.023513`
- `lag_15__CT_place_OBSERVATION`: contribution `-0.008265`
- `lag_00__T_kills_last_3s`: contribution `-0.005580`
- `lag_04__T_place_MINI`: contribution `-0.004787`
- `lag_00__kill_diff_last_3s`: contribution `-0.004675`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `218643`, seconds `76.50`, LSTM delta `+0.1092`

Top all feature movements:
- `lag_02__T_place_MINI`: contribution `+0.023513`
- `lag_01__T_place_MINI`: contribution `+0.014124`
- `lag_00__T_place_MINI`: contribution `-0.013080`
- `lag_13__T_place_CONTROL`: contribution `+0.005464`
- `lag_00__kill_diff_last_3s`: contribution `+0.004675`

Top utility-only movements:
- No utility movement among the top local contributors.
