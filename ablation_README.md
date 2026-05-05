# Ablation Osszefoglalo

Ez a jegyzet a ket uj ablation futast foglalja ossze:

- `artifacts/xgboost_ablation_reduced_no_utility_libdefaults`
- `artifacts/xgboost_ablation_reduced_with_utility_libdefaults`

A cel az volt, hogy megnezzuk:

- a utility feature-ok tenyleg adnak-e plusz informaciot
- vagy csak az eros, direkt nyeres-jelzo feature-ok mellett tunnek el

Ezert ebben a korben mindket modellbol kivettuk a legerosebb nem-utility blokkokat:

- `alive / hp / armor` jellegu feature-ok
- economy / equip jellegu feature-ok

Viszont bent hagytuk:

- recent combat feature-okat
- objective clue feature-okat
- pozicionalis feature-okat
- idobeli feature-okat

Igy a ket futas kozott a fo kulonbseg az volt, hogy:

- a `reduced_no_utility` modellnel a utility feature-ok is ki voltak dobva
- a `reduced_with_utility` modellnel a utility feature-ok bent maradtak

Mindket futas ugyanazzal a mintavetelezessel ment:

- `sample_csv_ratio = 0.3`
- `row_stride = 1`

Es ugyanazzal a harom explicit foparameterrel:

- `max_depth = 6`
- `learning_rate = 0.05`
- `n_estimators = 400`

A tobbi hyperparameterhez a script mar a library-default modot hasznalta.

## Futasok

### 1. Reduced No Utility

- output: `artifacts/xgboost_ablation_reduced_no_utility_libdefaults`
- feature count: `334`

Eredmenyek:

- train: accuracy `0.8275`, logloss `0.3914`, roc_auc `0.9190`
- valid: accuracy `0.7253`, logloss `0.5045`, roc_auc `0.8166`
- test: accuracy `0.7394`, logloss `0.4951`, roc_auc `0.8223`

### 2. Reduced With Utility

- output: `artifacts/xgboost_ablation_reduced_with_utility_libdefaults`
- feature count: `425`

Eredmenyek:

- train: accuracy `0.8479`, logloss `0.3648`, roc_auc `0.9324`
- valid: accuracy `0.7428`, logloss `0.4795`, roc_auc `0.8386`
- test: accuracy `0.7393`, logloss `0.4827`, roc_auc `0.8337`

## Kozvetlen Osszehasonlitas

`reduced_with_utility - reduced_no_utility`:

- train accuracy: `+0.0204`
- train logloss: `-0.0266`
- train roc_auc: `+0.0134`
- valid accuracy: `+0.0175`
- valid logloss: `-0.0250`
- valid roc_auc: `+0.0220`
- test accuracy: `-0.0001`
- test logloss: `-0.0124`
- test roc_auc: `+0.0114`

## Mit Jelent Ez

Ez a minta mar sokkal tisztabb, mint a korabbi teljes-feature-os utility vs no-utility osszevetes.

A fo megfigyeles:

- amikor kivettuk a nagyon eros `alive / hp / armor / economy / equip` blokkot
- a utility feature-ok egyertelmuen javitottak a modellt

Ez latszik abbol, hogy a `reduced_with_utility` jobb lett:

- valid accuracy-ben
- valid loglossban
- valid AUC-ben
- test loglossban
- test AUC-ben

A test accuracy gyakorlatilag ugyanaz maradt, de a `logloss` es a `roc_auc` erdemben javult.

Ez azt sugallja, hogy:

- a utility feature-ok tenyleg hordoznak hasznos jelet
- csak a teljes feature-keszletben ezt a jelet reszben elnyomjak az erosebb, direkt allapotjelzo feature-ok

## Mukodott-e Az Otlet

Rovid valasz:

- igen, mint diagnosztikai otlet mukodott
- nem, mint vegso teljesitmenyjavito modell-irany nem ez lett a gyoztes

Pontosabban:

- az otlet arra jo volt, hogy lathato legyen a utility marginalis hozzajarulasa
- ebben a redukalt setupban a utility hasznosnak bizonyult
- viszont az egesz redukalt modellcsalad gyengebb maradt, mint a teljes feature-keszletu korabbi modellek

## Osszevetes A Korabbi Baseline-okkal

`reduced_no_utility - baseline_no_utility`:

- train accuracy: `-0.0676`
- train logloss: `+0.0920`
- train roc_auc: `-0.0452`
- valid accuracy: `-0.0205`
- valid logloss: `+0.0332`
- valid roc_auc: `-0.0285`
- test accuracy: `-0.0071`
- test logloss: `+0.0222`
- test roc_auc: `-0.0212`

`reduced_with_utility - baseline_all_utility`:

- train accuracy: `-0.0497`
- train logloss: `+0.0665`
- train roc_auc: `-0.0331`
- valid accuracy: `-0.0063`
- valid logloss: `+0.0106`
- valid roc_auc: `-0.0084`
- test accuracy: `-0.0064`
- test logloss: `+0.0096`
- test roc_auc: `-0.0106`

Ez azt mutatja, hogy:

- a redukalt setup hasznos volt elemzesre
- de a teljes feature-keszlet tovabbra is jobb prediktiv modellnek

## Jelenlegi Kovetkeztetes

Most mar sokkal megalapozottabban lehet ezt mondani:

- a utility feature-ok nem feleslegesek
- valoban adnak plusz jelet
- de ez a plusz jel kisebb, mint a legerosebb allapot- es gazdasagi feature-ok hatasa

Maskepp fogalmazva:

- a utility nem a fo informacioforras
- hanem egy masodlagos, de valos hozzajarulo blokk

## Gyakorlati Tanulsag

Ha a cel a legjobb teljesitmeny, akkor:

- tovabbra is a teljes feature-keszletu modellek a jobbak

Ha a cel az ertelmezes, hogy a utility onmagaban mennyit tesz hozza, akkor:

- ez az ablation kor megerositette, hogy van ertelme a utilitynek
- csak a hatasa a teljes modellben reszben elfedodik az erosebb feature-ok miatt

## Vegso Osszegzes

Ez a ket futas alapjan a legkorrektebb allitas:

A utility feature-ok tenylegesen javitottak a redukalt modellen, tehat az otlet mukodott mint ellenorzo kiserlet, de a teljes feature-keszlethez kepest a redukalt modellek gyengebbek maradtak, vagyis a utility haszna valos, csak nem dominans a legerosebb nem-utility feature-okhoz kepest.

## Bovites: 50%-os Mintan

Az elso ket ablation futas utan ugyanazt a ket modellt lefuttattuk nagyobb mintan is:

- `artifacts/xgboost_ablation_reduced_no_utility_libdefaults_50pct`
- `artifacts/xgboost_ablation_reduced_with_utility_libdefaults_50pct`

Itt a beallitas ugyanaz maradt, csak:

- `sample_csv_ratio = 0.5`

## 50%-os Futasok

### Reduced No Utility 50%

- output: `artifacts/xgboost_ablation_reduced_no_utility_libdefaults_50pct`
- feature count: `334`

Eredmenyek:

- train: accuracy `0.7959`, logloss `0.4245`, roc_auc `0.8920`
- valid: accuracy `0.7408`, logloss `0.4843`, roc_auc `0.8349`
- test: accuracy `0.7397`, logloss `0.4908`, roc_auc `0.8299`

### Reduced With Utility 50%

- output: `artifacts/xgboost_ablation_reduced_with_utility_libdefaults_50pct`
- feature count: `425`

Eredmenyek:

- train: accuracy `0.8158`, logloss `0.3976`, roc_auc `0.9079`
- valid: accuracy `0.7463`, logloss `0.4682`, roc_auc `0.8465`
- test: accuracy `0.7473`, logloss `0.4773`, roc_auc `0.8404`

## Kozvetlen Osszehasonlitas 50%-on

`reduced_with_utility_50pct - reduced_no_utility_50pct`:

- train accuracy: `+0.0199`
- train logloss: `-0.0268`
- train roc_auc: `+0.0159`
- valid accuracy: `+0.0055`
- valid logloss: `-0.0161`
- valid roc_auc: `+0.0116`
- test accuracy: `+0.0076`
- test logloss: `-0.0135`
- test roc_auc: `+0.0105`

## 30% Vs 50%

### Reduced No Utility: 50% - 30%

- train accuracy: `-0.0315`
- train logloss: `+0.0330`
- train roc_auc: `-0.0271`
- valid accuracy: `+0.0155`
- valid logloss: `-0.0202`
- valid roc_auc: `+0.0182`
- test accuracy: `+0.0003`
- test logloss: `-0.0044`
- test roc_auc: `+0.0076`

Olvasat:

- a nagyobb mintan a no-utility modell trainen gyengebb lett
- viszont validon es teszten javult
- ez arra utal, hogy a 50%-os minta kevesbe engedte railleszkedni a modellt a train adatra, mikozben jobb generalizaciot adott

### Reduced With Utility: 50% - 30%

- train accuracy: `-0.0320`
- train logloss: `+0.0328`
- train roc_auc: `-0.0245`
- valid accuracy: `+0.0035`
- valid logloss: `-0.0114`
- valid roc_auc: `+0.0079`
- test accuracy: `+0.0080`
- test logloss: `-0.0054`
- test roc_auc: `+0.0066`

Olvasat:

- ugyanez a minta latszik a utilitys modellen is
- a train score visszaesett, de a valid es test javult
- vagyis a nagyobb minta itt is jobb altalanositast adott

## Mit Jelent A 30% Es 50% Egyutt

Ez az uj kor erositi a korabbi kovetkeztetest.

30%-on azt lattuk, hogy:

- a utility javitotta a redukalt modellt
- de a test accuracy meg alig mozdult

50%-on viszont mar ez latszik:

- a utility nemcsak loglossban es AUC-ben segit
- hanem test accuracyben is lathato pluszt ad

Ez fontos, mert azt mutatja, hogy a utilitys elony:

- nem tunt el nagyobb mintan
- hanem meg stabilabbnak latszik

## Frissitett Kovetkeztetes

A mostani 30%-os es 50%-os ablation korok alapjan a legkorrektebb allitas mar ez:

- a utility feature-ok valoszinuleg tenyleg hordoznak onallo, hasznos informaciot
- ez a plusz informacio jobban latszik, ha a legerosebb `alive / hp / armor / economy / equip` feature-okat kivesszuk
- a nagyobb, 50%-os mintan a utilitys elony nem gyengult, hanem inkabb megerosodott

Ugyanakkor tovabbra is igaz:

- a teljes feature-keszletu modellek osszteljesitmenye meg mindig jobb
- ezert a utility jelenleg nem helyettesiti a legerosebb allapotjelzo feature-okat
- hanem kiegeszito, masodlagos, de valos hozzajarulo blokk

## Friss Vegso Osszegzes

Az ablation otlet nemcsak egyszeri, 30%-os mintan mukodott, hanem 50%-os mintan is megerositest kapott: a utility feature-ok a redukalt modellekben kovetkezetesen javitottak a generalizaciot, vagyis a hasznuk valos, csak a teljes feature-keszletben a hatasukat reszben elfedik a meg erosebb nem-utility feature-ok.


accuracy precision f1 score 
recall
brier score
confusion matrix
calibration curve

accurac ynincs nagy elteeres kulonbsegmeres mashogy (osszehasonlitas ket podell predictelt win probalilityjeit) Delta win probability)

filter mikor aktivak
mikor jo a 

ott nagyobb delta ahol 