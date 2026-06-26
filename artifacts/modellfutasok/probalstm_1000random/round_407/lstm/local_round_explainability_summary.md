# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `21541`, seconds `123.50`, LSTM `0.1711`, delta `+0.1133`
- tick `21637`, seconds `125.00`, LSTM `0.4011`, delta `+0.1116`
- tick `21605`, seconds `124.50`, LSTM `0.2895`, delta `+0.0719`
- tick `21669`, seconds `125.50`, LSTM `0.4648`, delta `+0.0637`
- tick `21989`, seconds `130.50`, LSTM `0.5746`, delta `+0.0592`
- tick `21701`, seconds `126.00`, LSTM `0.5194`, delta `+0.0546`
- tick `21765`, seconds `127.00`, LSTM `0.4656`, delta `-0.0535`
- tick `21573`, seconds `124.00`, LSTM `0.2176`, delta `+0.0465`
- tick `21893`, seconds `129.00`, LSTM `0.4986`, delta `+0.0461`
- tick `21861`, seconds `128.50`, LSTM `0.4525`, delta `+0.0443`

## Top 15 local ridge features

- `lag_15__CT_place_SIDEALLEY`: coefficient `0.002550`, |coef| `0.002550`
- `lag_03__CT_place_BACKALLEY`: coefficient `0.002394`, |coef| `0.002394`
- `lag_11__CT_place_BACKALLEY`: coefficient `0.001919`, |coef| `0.001919`
- `lag_12__CT_place_BACKALLEY`: coefficient `0.001912`, |coef| `0.001912`
- `lag_04__CT_place_BACKALLEY`: coefficient `0.001875`, |coef| `0.001875`
- `lag_02__CT_place_BACKALLEY`: coefficient `0.001837`, |coef| `0.001837`
- `lag_01__CT_place_BACKALLEY`: coefficient `0.001575`, |coef| `0.001575`
- `lag_13__CT_place_BACKALLEY`: coefficient `0.001298`, |coef| `0.001298`
- `lag_14__CT_place_SIDEALLEY`: coefficient `0.001276`, |coef| `0.001276`
- `lag_06__CT_place_BACKALLEY`: coefficient `0.001156`, |coef| `0.001156`
- `lag_15__CT_place_BACKALLEY`: coefficient `0.001112`, |coef| `0.001112`
- `lag_07__T_place_TRUCK`: coefficient `-0.001061`, |coef| `0.001061`
- `lag_14__CT_place_BACKALLEY`: coefficient `0.001033`, |coef| `0.001033`
- `lag_05__CT_place_BACKALLEY`: coefficient `0.001011`, |coef| `0.001011`
- `lag_11__CT_place_SIDEALLEY`: coefficient `0.000914`, |coef| `0.000914`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.000467` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000454` (lowers CT win probability)
- `lag_01__CT3__flash`: coefficient `0.000395` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000341` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000329` (lowers CT win probability)
- `lag_07__T2__flash`: coefficient `-0.000309` (lowers CT win probability)
- `lag_02__CT3__flash`: coefficient `0.000301` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.000285` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000275` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000259` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SIDEALLEY`: coefficient `0.002550` (raises CT win probability)
- `lag_03__CT_place_BACKALLEY`: coefficient `0.002394` (raises CT win probability)
- `lag_11__CT_place_BACKALLEY`: coefficient `0.001919` (raises CT win probability)
- `lag_12__CT_place_BACKALLEY`: coefficient `0.001912` (raises CT win probability)
- `lag_04__CT_place_BACKALLEY`: coefficient `0.001875` (raises CT win probability)
- `lag_02__CT_place_BACKALLEY`: coefficient `0.001837` (raises CT win probability)
- `lag_01__CT_place_BACKALLEY`: coefficient `0.001575` (raises CT win probability)
- `lag_13__CT_place_BACKALLEY`: coefficient `0.001298` (raises CT win probability)
- `lag_14__CT_place_SIDEALLEY`: coefficient `0.001276` (raises CT win probability)
- `lag_06__CT_place_BACKALLEY`: coefficient `0.001156` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `21541`, seconds `123.50`, LSTM delta `+0.1133`

Top all feature movements:
- `lag_15__CT_place_SIDEALLEY`: contribution `+0.046522`
- `lag_01__CT_place_BACKALLEY`: contribution `+0.023604`
- `lag_08__T_place_TRUCK`: contribution `+0.014943`
- `lag_04__T_place_TRUCK`: contribution `+0.012919`
- `lag_06__T_place_TRUCK`: contribution `-0.012489`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21637`, seconds `125.00`, LSTM delta `+0.1116`

Top all feature movements:
- `lag_04__CT_place_BACKALLEY`: contribution `+0.028105`
- `lag_07__T_place_TRUCK`: contribution `+0.018431`
- `lag_14__T_place_TRUCK`: contribution `+0.009844`
- `lag_04__CT_place_UNDERPASS`: contribution `+0.003904`
- `lag_11__T_place_TRUCK`: contribution `-0.003897`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21605`, seconds `124.50`, LSTM delta `+0.0719`

Top all feature movements:
- `lag_03__CT_place_BACKALLEY`: contribution `+0.035884`
- `lag_08__T_place_TRUCK`: contribution `-0.014943`
- `lag_06__T_place_TRUCK`: contribution `+0.012489`
- `lag_10__T_place_TRUCK`: contribution `+0.005303`
- `lag_03__CT_place_UNDERPASS`: contribution `+0.002930`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21669`, seconds `125.50`, LSTM delta `+0.0637`

Top all feature movements:
- `lag_05__CT_place_BACKALLEY`: contribution `+0.015155`
- `lag_08__T_place_TRUCK`: contribution `+0.014943`
- `lag_15__T_place_TRUCK`: contribution `+0.010767`
- `lag_10__T_place_TRUCK`: contribution `-0.005303`
- `lag_12__CT_place_SHOP`: contribution `+0.002227`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21989`, seconds `130.50`, LSTM delta `+0.0592`

Top all feature movements:
- `lag_15__CT_place_BACKALLEY`: contribution `+0.016669`
- `lag_00__CT_place_SIDEALLEY`: contribution `+0.011997`
- `lag_08__CT_place_BACKALLEY`: contribution `+0.008868`
- `lag_06__CT_place_SIDEALLEY`: contribution `+0.006250`
- `lag_15__CT_place_UNDERPASS`: contribution `+0.003155`

Top utility-only movements:
- No utility movement among the top local contributors.
