# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `13180`, seconds `48.00`, LSTM `0.5125`, delta `-0.2637`
- tick `13244`, seconds `49.00`, LSTM `0.7667`, delta `+0.2445`
- tick `13308`, seconds `50.00`, LSTM `0.6043`, delta `-0.2191`
- tick `13052`, seconds `46.00`, LSTM `0.6811`, delta `+0.1611`
- tick `13084`, seconds `46.50`, LSTM `0.7954`, delta `+0.1143`
- tick `12572`, seconds `38.50`, LSTM `0.5117`, delta `+0.1131`
- tick `12316`, seconds `34.50`, LSTM `0.3549`, delta `+0.0789`
- tick `12156`, seconds `32.00`, LSTM `0.4401`, delta `-0.0750`
- tick `12188`, seconds `32.50`, LSTM `0.3749`, delta `-0.0652`
- tick `12348`, seconds `35.00`, LSTM `0.4165`, delta `+0.0616`

## Top 15 local ridge features

- `lag_10__CT_place_VENDING`: coefficient `0.001931`, |coef| `0.001931`
- `lag_03__CT_place_VENDING`: coefficient `0.001827`, |coef| `0.001827`
- `lag_00__CT_place_LOBBY`: coefficient `0.001742`, |coef| `0.001742`
- `lag_08__T_place_HUT`: coefficient `0.001709`, |coef| `0.001709`
- `lag_12__CT_place_VENDING`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_15__CT_place_TROPHY`: coefficient `0.001665`, |coef| `0.001665`
- `lag_06__T_place_VENTS`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_00__T_place_HUT`: coefficient `0.001599`, |coef| `0.001599`
- `lag_08__CT_place_TROPHY`: coefficient `0.001575`, |coef| `0.001575`
- `lag_05__T_place_VENTS`: coefficient `-0.001455`, |coef| `0.001455`
- `lag_06__CT_place_HELL`: coefficient `0.001448`, |coef| `0.001448`
- `lag_13__CT_place_CONTROL`: coefficient `0.001412`, |coef| `0.001412`
- `lag_04__T_place_VENTS`: coefficient `-0.001398`, |coef| `0.001398`
- `lag_00__CT_place_VENDING`: coefficient `-0.001386`, |coef| `0.001386`
- `lag_00__T_place_VENTS`: coefficient `-0.001375`, |coef| `0.001375`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `0.000446` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.000423` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000404` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000340` (lowers CT win probability)
- `lag_08__T4__molly`: coefficient `-0.000324` (lowers CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.000323` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `0.000319` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `0.000310` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000297` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000283` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_VENDING`: coefficient `0.001931` (raises CT win probability)
- `lag_03__CT_place_VENDING`: coefficient `0.001827` (raises CT win probability)
- `lag_00__CT_place_LOBBY`: coefficient `0.001742` (raises CT win probability)
- `lag_08__T_place_HUT`: coefficient `0.001709` (raises CT win probability)
- `lag_12__CT_place_VENDING`: coefficient `-0.001685` (lowers CT win probability)
- `lag_15__CT_place_TROPHY`: coefficient `0.001665` (raises CT win probability)
- `lag_06__T_place_VENTS`: coefficient `-0.001611` (lowers CT win probability)
- `lag_00__T_place_HUT`: coefficient `0.001599` (raises CT win probability)
- `lag_08__CT_place_TROPHY`: coefficient `0.001575` (raises CT win probability)
- `lag_05__T_place_VENTS`: coefficient `-0.001455` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `13180`, seconds `48.00`, LSTM delta `-0.2637`

Top all feature movements:
- `lag_03__CT_place_VENDING`: contribution `-0.031313`
- `lag_08__CT_place_TROPHY`: contribution `-0.023260`
- `lag_08__CT_place_VENDING`: contribution `-0.022475`
- `lag_13__CT_place_TROPHY`: contribution `-0.016703`
- `lag_08__T_place_HUT`: contribution `-0.015933`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13244`, seconds `49.00`, LSTM delta `+0.2445`

Top all feature movements:
- `lag_10__CT_place_VENDING`: contribution `+0.033091`
- `lag_15__CT_place_TROPHY`: contribution `+0.024584`
- `lag_00__T_place_HUT`: contribution `+0.014905`
- `lag_05__CT_place_VENDING`: contribution `+0.010659`
- `lag_15__CT_place_CONTROL`: contribution `+0.010424`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13308`, seconds `50.00`, LSTM delta `-0.2191`

Top all feature movements:
- `lag_12__CT_place_VENDING`: contribution `-0.028872`
- `lag_07__CT_place_VENDING`: contribution `-0.021434`
- `lag_12__CT_place_TROPHY`: contribution `-0.015915`
- `lag_00__CT_place_LOBBY`: contribution `-0.014258`
- `lag_05__T_place_HUT`: contribution `-0.009185`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13052`, seconds `46.00`, LSTM delta `+0.1611`

Top all feature movements:
- `lag_04__CT_place_VENDING`: contribution `+0.020442`
- `lag_08__T_place_HUT`: contribution `+0.015933`
- `lag_09__CT_place_TROPHY`: contribution `+0.015190`
- `lag_04__CT_place_TROPHY`: contribution `+0.012990`
- `lag_09__CT_place_CONTROL`: contribution `+0.010194`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13084`, seconds `46.50`, LSTM delta `+0.1143`

Top all feature movements:
- `lag_00__CT_place_VENDING`: contribution `+0.023758`
- `lag_05__CT_place_TROPHY`: contribution `+0.017648`
- `lag_00__CT_place_LOBBY`: contribution `+0.014258`
- `lag_10__CT_place_CONTROL`: contribution `+0.011187`
- `lag_05__CT_place_VENDING`: contribution `-0.010659`

Top utility-only movements:
- No utility movement among the top local contributors.
