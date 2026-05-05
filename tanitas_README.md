# Tanitas Osszefoglalo

Ez a jegyzet a negy eddigi XGBoost futast foglalja ossze:

- `artifacts/xgboost_baseline_no_utility`
- `artifacts/xgboost_baseline_all_utility`
- `artifacts/xgboost_tuned_no_utility_regularized`
- `artifacts/xgboost_tuned_all_utility_regularized`

Minden futas ugyanazon az alap adatlogikan ment:

- csak teljes CSV-k, `-p1/-p2` nelkul
- match-szintu train/valid/test split
- `sample_csv_ratio = 0.3`
- `row_stride = 1`
- csak numerikus feature-ok
- `tick` nincs bent

## Beallitasok

### 1. Baseline No Utility

- output: `artifacts/xgboost_baseline_no_utility`
- utility feature-ok ki voltak dobva
- feature count: `397`
- parameter:
  - `max_depth = 6`
  - `n_estimators = 400`
  - `learning_rate = 0.05`
  - `min_child_weight = 5`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `reg_lambda = 2.0`
  - `reg_alpha = 0.0`

### 2. Baseline All Utility

- output: `artifacts/xgboost_baseline_all_utility`
- utility feature-ok bent maradtak
- feature count: `488`
- parameter:
  - `max_depth = 6`
  - `n_estimators = 400`
  - `learning_rate = 0.05`
  - `min_child_weight = 5`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `reg_lambda = 2.0`
  - `reg_alpha = 0.0`

### 3. Tuned No Utility Regularized

- output: `artifacts/xgboost_tuned_no_utility_regularized`
- utility feature-ok ki voltak dobva
- feature count: `397`
- parameter:
  - `max_depth = 4`
  - `n_estimators = 600`
  - `learning_rate = 0.03`
  - `min_child_weight = 10`
  - `subsample = 0.7`
  - `colsample_bytree = 0.7`
  - `reg_lambda = 4.0`
  - `reg_alpha = 0.5`

### 4. Tuned All Utility Regularized

- output: `artifacts/xgboost_tuned_all_utility_regularized`
- utility feature-ok bent maradtak
- feature count: `488`
- parameter:
  - `max_depth = 4`
  - `n_estimators = 600`
  - `learning_rate = 0.03`
  - `min_child_weight = 10`
  - `subsample = 0.7`
  - `colsample_bytree = 0.7`
  - `reg_lambda = 4.0`
  - `reg_alpha = 0.5`

## Eredmenyek

### Baseline No Utility

- train: accuracy `0.8951`, logloss `0.2994`, roc_auc `0.9642`
- valid: accuracy `0.7458`, logloss `0.4713`, roc_auc `0.8452`
- test: accuracy `0.7465`, logloss `0.4729`, roc_auc `0.8435`

### Baseline All Utility

- train: accuracy `0.8975`, logloss `0.2982`, roc_auc `0.9654`
- valid: accuracy `0.7491`, logloss `0.4689`, roc_auc `0.8470`
- test: accuracy `0.7457`, logloss `0.4731`, roc_auc `0.8443`

### Tuned No Utility Regularized

- train: accuracy `0.8101`, logloss `0.3924`, roc_auc `0.9037`
- valid: accuracy `0.7509`, logloss `0.4626`, roc_auc `0.8497`
- test: accuracy `0.7536`, logloss `0.4643`, roc_auc `0.8486`

### Tuned All Utility Regularized

- train: accuracy `0.8101`, logloss `0.3921`, roc_auc `0.9042`
- valid: accuracy `0.7511`, logloss `0.4622`, roc_auc `0.8500`
- test: accuracy `0.7518`, logloss `0.4648`, roc_auc `0.8485`

## Gyors Ranglista

Teszt alapjan a jelenlegi sorrend:

1. `tuned_no_utility_regularized`
   - accuracy `0.7536`
   - logloss `0.4643`
   - roc_auc `0.8486`
2. `tuned_all_utility_regularized`
   - accuracy `0.7518`
   - logloss `0.4648`
   - roc_auc `0.8485`
3. `baseline_all_utility`
   - accuracy `0.7457`
   - logloss `0.4731`
   - roc_auc `0.8443`
4. `baseline_no_utility`
   - accuracy `0.7465`
   - logloss `0.4729`
   - roc_auc `0.8435`

## Kozvetlen Osszehasonlitas

### Baseline szinten: `all_utility - no_utility`

- train accuracy: `+0.0024`
- train logloss: `-0.0012`
- train roc_auc: `+0.0012`
- valid accuracy: `+0.0033`
- valid logloss: `-0.0024`
- valid roc_auc: `+0.0019`
- test accuracy: `-0.0008`
- test logloss: `+0.0001`
- test roc_auc: `+0.0008`

Olvasat:

- baseline beallitas mellett a utility feature-ok validon hasznosnak tuntek
- teszten viszont nem adtak tiszta, stabil elonyt

### Tuned szinten: `all_utility - no_utility`

- train accuracy: `-0.0000`
- train logloss: `-0.0003`
- train roc_auc: `+0.0004`
- valid accuracy: `+0.0002`
- valid logloss: `-0.0003`
- valid roc_auc: `+0.0003`
- test accuracy: `-0.0017`
- test logloss: `+0.0005`
- test roc_auc: `-0.0001`

Olvasat:

- a tuned beallitasnal a ket modell mar majdnem egyforma
- validon az `all_utility` picit jobb
- teszten viszont a `no_utility` lett hajszallal jobb
- a kulonbseg nagyon kicsi, de jelenleg a teszt gyoztes a `tuned_no_utility`

### No Utility iranyban: `tuned - baseline`

`tuned_no_utility - baseline_no_utility`:

- train accuracy: `-0.0850`
- train logloss: `+0.0930`
- train roc_auc: `-0.0605`
- valid accuracy: `+0.0051`
- valid logloss: `-0.0088`
- valid roc_auc: `+0.0045`
- test accuracy: `+0.0070`
- test logloss: `-0.0086`
- test roc_auc: `+0.0051`

Olvasat:

- a train teljesitmeny sokkal gyengebb lett
- viszont validon es teszten minden fontos metrika javult
- ez eros jel arra, hogy a regularizalt tuning csokkentette a tulilleszkedest

### All Utility iranyban: `tuned - baseline`

`tuned_all_utility - baseline_all_utility`:

- train accuracy: `-0.0875`
- train logloss: `+0.0939`
- train roc_auc: `-0.0613`
- valid accuracy: `+0.0020`
- valid logloss: `-0.0067`
- valid roc_auc: `+0.0029`
- test accuracy: `+0.0061`
- test logloss: `-0.0082`
- test roc_auc: `+0.0042`

Olvasat:

- itt is jelentosen visszaesett a train score
- megis jobb lett a valid es a test
- vagyis utilityvel is mukodik a regularizalt tuning, csak a vegso teszt elony picit kisebb lett, mint `no_utility` mellett

## Mit Jelentenek Ezek Egyutt

A negy futas alapjan a legerosebb minta most ez:

- nem a nagyobb train score adta a jobb modellt
- a regularizalt, sekelyebb, lassabban tanulo beallitasok jobb generalizaciot adtak
- a baseline modellek jobban railleszkedtek a train adatra
- a tuned modellek kevesbe illeszkedtek ra trainen, de jobbak lettek validon es teszten

Ez klasszikusan arra utal, hogy a tuning fo hatasa nem az volt, hogy "erossebb" modellt csinalt, hanem az, hogy visszafogta a tulilleszkedest.

## Minek Koszonheto Valoszinuleg A Javulas

### Hyperparameter oldalrol

Valoszinuleg ezek segitettek a legtobbet:

- `max_depth = 4`
  - sekelyebb fak
  - kevesebb nagyon specifikus, zajra illeszkedo split
- `min_child_weight = 10`
  - nehezebben hoz letre apro, bizonytalan leveleket
- `subsample = 0.7`
  - minden fa kevesebb mintan tanul, ami regularizal
- `colsample_bytree = 0.7`
  - kevesebb feature-bol epit egy-egy fat, ami csokkenti a tullovest
- `reg_lambda = 4.0` es `reg_alpha = 0.5`
  - erosebb sulyregularizacio
- `learning_rate = 0.03` + `n_estimators = 600`
  - lassabb, fokozatosabb tanulas
  - kisebb az eselye, hogy agresszivan ratanul a train mintazataira

### Feature oldalrol

A utility feature-okrol most ovatosabb, de tisztabb a kep:

- baseline mellett validon adtak pluszt, de teszten nem volt stabil az elony
- tuned mellett a utility es a no_utility mar majdnem ugyanott van
- ez arra utalhat, hogy a utility feature-ok tartalmaznak informaciot
- de a mostani 30%-os mintan ez az informacio nem eleg eros ahhoz, hogy robusztusan elvigye a no_utility verzio folott a tesztet

Maskepp fogalmazva:

- a fo, megbizhato javulast most a hyperparameter tuning hozta
- nem a utility feature-ok hoztak a legnagyobb nyeresegnek latszo lepeset

## Jelenlegi Legkorrektebb Kovetkeztetes

Ha most egy ovatos, szakmailag vedheto mondatban kellene osszefoglalni:

A legjobb jelenlegi modell a `tuned_no_utility_regularized`, es a legerosebb javulas valoszinuleg a szigorubb regularizacionak es a jobb generalizacionak koszonheto, nem annak, hogy a utility feature-ok onmagukban egyertelmuen megoldottak volna a feladatot.

## Mit Erdemes Innen Tovabbvinni

A mostani eredmenyek alapjan a legjobb kovetkezo iranyok:

- `early stopping`, hogy a fa-szam ne fix legyen, hanem validacio alapjan alljon meg
- `random search`, hogy a regularizalt parameterterben ne kezzel csak egy pontot nezzunk meg
- nagyobb minta vagy teljes adat, hogy kideruljon a utility feature-ok elonye stabilizalodik-e
- kesobb akar blokkos utility ablation, ha azt akarjuk megerteni, melyik utility-csoport hoz tenyleges hasznot
