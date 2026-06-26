# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `23`

## Largest probability jumps

- tick `213468`, seconds `95.00`, LSTM `0.6891`, delta `+0.2246`
- tick `215516`, seconds `127.00`, LSTM `0.0782`, delta `-0.2093`
- tick `213596`, seconds `97.00`, LSTM `0.4882`, delta `-0.1800`
- tick `211740`, seconds `68.00`, LSTM `0.5523`, delta `+0.1711`
- tick `210908`, seconds `55.00`, LSTM `0.3964`, delta `-0.1668`
- tick `210364`, seconds `46.50`, LSTM `0.4214`, delta `-0.1537`
- tick `215292`, seconds `123.50`, LSTM `0.2907`, delta `+0.1078`
- tick `210812`, seconds `53.50`, LSTM `0.5603`, delta `+0.1036`
- tick `213916`, seconds `102.00`, LSTM `0.2639`, delta `+0.0905`
- tick `213884`, seconds `101.50`, LSTM `0.1733`, delta `-0.0793`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004054`, |coef| `0.004054`
- `lag_00__damage_diff_last_5s`: coefficient `0.003796`, |coef| `0.003796`
- `lag_00__CT_kills_last_3s`: coefficient `0.002584`, |coef| `0.002584`
- `lag_00__T_kills_last_3s`: coefficient `-0.002500`, |coef| `0.002500`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002479`, |coef| `0.002479`
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `0.002207`, |coef| `0.002207`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002101`, |coef| `0.002101`
- `lag_00__CT_damage_last_5s`: coefficient `0.002031`, |coef| `0.002031`
- `lag_10__T_place_CONNECTOR`: coefficient `-0.001920`, |coef| `0.001920`
- `lag_13__CT2__is_scoped`: coefficient `0.001909`, |coef| `0.001909`
- `lag_07__T_place_MAIN`: coefficient `-0.001831`, |coef| `0.001831`
- `lag_00__T_damage_last_5s`: coefficient `-0.001801`, |coef| `0.001801`
- `lag_00__CT_place_BACKOFB`: coefficient `0.001799`, |coef| `0.001799`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001786`, |coef| `0.001786`
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001780`, |coef| `0.001780`

## Top 10 utility ridge features

- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001780` (lowers CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.001466` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001175` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001149` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001074` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000967` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000945` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000871` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000777` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.000724` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004054` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003796` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002584` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002500` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002479` (lowers CT win probability)
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `0.002207` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002101` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002031` (raises CT win probability)
- `lag_10__T_place_CONNECTOR`: coefficient `-0.001920` (lowers CT win probability)
- `lag_13__CT2__is_scoped`: coefficient `0.001909` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `213468`, seconds `95.00`, LSTM delta `+0.2246`

Top all feature movements:
- `lag_07__T_place_MAIN`: contribution `+0.011840`
- `lag_00__kill_diff_last_3s`: contribution `+0.009757`
- `lag_10__T_place_CONNECTOR`: contribution `+0.009296`
- `lag_00__damage_diff_last_5s`: contribution `+0.008564`
- `lag_03__T_place_CONNECTOR`: contribution `+0.008130`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `215516`, seconds `127.00`, LSTM delta `-0.2093`

Top all feature movements:
- `lag_10__CT_place_LOWERTUNNEL`: contribution `-0.016222`
- `lag_13__CT2__is_scoped`: contribution `-0.011686`
- `lag_00__kill_diff_last_3s`: contribution `-0.009757`
- `lag_00__T_shots_fired_sum`: contribution `-0.009293`
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.009207`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.009207`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.006222`

### tick `213596`, seconds `97.00`, LSTM delta `-0.1800`

Top all feature movements:
- `lag_00__CT_place_BACKOFB`: contribution `-0.010269`
- `lag_00__kill_diff_last_3s`: contribution `-0.009757`
- `lag_00__damage_diff_last_5s`: contribution `-0.008564`
- `lag_00__T_kills_last_3s`: contribution `-0.007920`
- `lag_02__CT_place_PALACEINTERIOR`: contribution `-0.006500`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.001919`

### tick `211740`, seconds `68.00`, LSTM delta `+0.1711`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009757`
- `lag_00__T_shots_fired_sum`: contribution `+0.009293`
- `lag_00__CT_kills_last_3s`: contribution `+0.007461`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007297`
- `lag_15__T_place_STREET`: contribution `+0.005114`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `210908`, seconds `55.00`, LSTM delta `-0.1668`

Top all feature movements:
- `lag_03__CT_place_BRICKS`: contribution `-0.016766`
- `lag_04__CT_place_BRICKS`: contribution `-0.013305`
- `lag_00__kill_diff_last_3s`: contribution `-0.009757`
- `lag_00__T_shots_fired_sum`: contribution `-0.009293`
- `lag_00__damage_diff_last_5s`: contribution `-0.008564`

Top utility-only movements:
- No utility movement among the top local contributors.
