# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `9`

## Largest probability jumps

- tick `69400`, seconds `51.50`, LSTM `0.4030`, delta `-0.2224`
- tick `68856`, seconds `43.00`, LSTM `0.6045`, delta `-0.2042`
- tick `69592`, seconds `54.50`, LSTM `0.0241`, delta `-0.1250`
- tick `69432`, seconds `52.00`, LSTM `0.2926`, delta `-0.1104`
- tick `69464`, seconds `52.50`, LSTM `0.2130`, delta `-0.0796`
- tick `67416`, seconds `20.50`, LSTM `0.7501`, delta `+0.0773`
- tick `68824`, seconds `42.50`, LSTM `0.8087`, delta `+0.0562`
- tick `66200`, seconds `1.50`, LSTM `0.6534`, delta `+0.0447`
- tick `68632`, seconds `39.50`, LSTM `0.7645`, delta `-0.0428`
- tick `67896`, seconds `28.00`, LSTM `0.7875`, delta `+0.0367`

## Top 15 local ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.003214`, |coef| `0.003214`
- `lag_01__CT_place_UPPERPARK`: coefficient `0.002658`, |coef| `0.002658`
- `lag_08__T_place_CONNECTOR`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_00__CT1__utility_total`: coefficient `0.001825`, |coef| `0.001825`
- `lag_07__CT_shots_fired_sum`: coefficient `0.001741`, |coef| `0.001741`
- `lag_00__CT1__flash`: coefficient `0.001739`, |coef| `0.001739`
- `lag_00__T_kills_last_3s`: coefficient `-0.001736`, |coef| `0.001736`
- `lag_02__CT_place_UPPERPARK`: coefficient `0.001733`, |coef| `0.001733`
- `lag_00__T_damage_last_5s`: coefficient `-0.001630`, |coef| `0.001630`
- `lag_00__kill_diff_last_3s`: coefficient `0.001626`, |coef| `0.001626`
- `lag_13__CT_place_CANAL`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_09__T_place_CONNECTOR`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_00__damage_diff_last_5s`: coefficient `0.001567`, |coef| `0.001567`
- `lag_03__CT_place_UPPERPARK`: coefficient `0.001493`, |coef| `0.001493`
- `lag_15__CT_place_FOUNTAIN`: coefficient `0.001460`, |coef| `0.001460`

## Top 10 utility ridge features

- `lag_00__CT1__utility_total`: coefficient `0.001825` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001739` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.001210` (raises CT win probability)
- `lag_01__CT1__utility_total`: coefficient `0.001131` (raises CT win probability)
- `lag_01__CT1__flash`: coefficient `0.001064` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001053` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001023` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000921` (raises CT win probability)
- `lag_13__T2__smoke`: coefficient `0.000801` (raises CT win probability)
- `lag_02__CT1__utility_total`: coefficient `0.000773` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.003214` (raises CT win probability)
- `lag_01__CT_place_UPPERPARK`: coefficient `0.002658` (raises CT win probability)
- `lag_08__T_place_CONNECTOR`: coefficient `-0.002018` (lowers CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `0.001741` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001736` (lowers CT win probability)
- `lag_02__CT_place_UPPERPARK`: coefficient `0.001733` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001630` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001626` (raises CT win probability)
- `lag_13__CT_place_CANAL`: coefficient `-0.001600` (lowers CT win probability)
- `lag_09__T_place_CONNECTOR`: coefficient `-0.001600` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `69400`, seconds `51.50`, LSTM delta `-0.2224`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.033810`
- `lag_01__CT_place_UPPERPARK`: contribution `-0.018922`
- `lag_08__T_place_CONNECTOR`: contribution `-0.009774`
- `lag_00__CT1__utility_total`: contribution `-0.006850`
- `lag_00__CT1__flash`: contribution `-0.006225`

Top utility-only movements:
- `lag_00__CT1__utility_total`: contribution `-0.006850`
- `lag_00__CT1__flash`: contribution `-0.006225`
- `lag_00__CT1__molly`: contribution `-0.003011`

### tick `68856`, seconds `43.00`, LSTM delta `-0.2042`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.033810`
- `lag_07__CT_shots_fired_sum`: contribution `-0.021771`
- `lag_01__CT_place_BRIDGE`: contribution `-0.013045`
- `lag_07__CT3__shots_fired`: contribution `-0.012701`
- `lag_13__CT_place_CANAL`: contribution `-0.009727`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `69592`, seconds `54.50`, LSTM delta `-0.1250`

Top all feature movements:
- `lag_01__T_place_FOUNTAIN`: contribution `-0.010335`
- `lag_07__CT_place_FOUNTAIN`: contribution `-0.008376`
- `lag_07__CT_place_UPPERPARK`: contribution `-0.005594`
- `lag_01__T_place_UPPERPARK`: contribution `-0.004882`
- `lag_01__T_place_LOWERPARK`: contribution `-0.004807`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.002362`
- `lag_06__CT1__utility_total`: contribution `-0.002289`
- `lag_06__CT1__flash`: contribution `-0.002045`

### tick `69432`, seconds `52.00`, LSTM delta `-0.1104`

Top all feature movements:
- `lag_02__CT_place_UPPERPARK`: contribution `-0.012335`
- `lag_02__CT_place_FOUNTAIN`: contribution `-0.008173`
- `lag_09__T_place_CONNECTOR`: contribution `-0.007750`
- `lag_06__T5__duck_amount`: contribution `-0.004762`
- `lag_04__CT1__is_scoped`: contribution `+0.004751`

Top utility-only movements:
- `lag_01__CT1__utility_total`: contribution `-0.004244`
- `lag_01__CT1__flash`: contribution `-0.003807`

### tick `69464`, seconds `52.50`, LSTM delta `-0.0796`

Top all feature movements:
- `lag_03__CT_place_UPPERPARK`: contribution `-0.010630`
- `lag_02__CT_place_FOUNTAIN`: contribution `+0.008173`
- `lag_01__CT_place_WATER`: contribution `-0.007936`
- `lag_03__CT_place_FOUNTAIN`: contribution `-0.006004`
- `lag_10__T_place_CONNECTOR`: contribution `-0.005293`

Top utility-only movements:
- `lag_02__CT1__utility_total`: contribution `-0.002901`
- `lag_02__CT1__flash`: contribution `-0.002595`
