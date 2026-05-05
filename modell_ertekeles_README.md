# Modell Ertekeles Es Feature Kovetkeztetesek

Ez a jegyzet osszefoglalja:

- milyen metrikakkal ertekeljuk az XGBoost modelleket
- mit jelentenek ezek a metrikak a CT win probability feladatban
- melyik eddigi modellfutas tunik a legjobbnak
- mit lattunk a `place` es `weapon` kategorias feature-okrol
- hogyan erdemes tovabb vizsgalni a modelleket

## Feladat

A modell celja binaris klasszifikacio:

- label: `ct_win`
- `1`: CT nyeri a kort
- `0`: T nyeri a kort

A modell nemcsak vegso 0/1 dontest ad, hanem valoszinuseget is:

- `predict_proba[:, 1]`
- ez ertelmezheto ugy, mint becsult CT win probability

Ezert ketfele ertekeles fontos:

- klasszifikacios teljesitmeny: jol talalja-e el a nyertest `0.5` threshold mellett
- probability teljesitmeny: jo, kalibralt es hasznos valoszinusegeket ad-e

## Train / Valid / Test Jelentese

### Train

Ezen tanul a modell.

Ha a train score nagyon jo, de valid/test gyengebb, akkor a modell valoszinuleg tulilleszkedik.

### Valid

Ezen valasztunk modellt es hyperparametert.

Fontos: a valid eredmeny alapjan szabad donteni, de ha tul sokszor nezzuk es ehhez igazitjuk a modelleket, akkor a validra is ra lehet illeszkedni.

### Test

Ez a legfontosabb vegso ellenorzes.

A test eredmenyt ugy kell kezelni, mint egy eddig nem latott meccs/minta becsleset. Ha a valid es test kozel vannak egymashoz, az jo jel.

## Accuracy

Az accuracy azt meri, hogy a `0.5` threshold mellett a modell hany szazalekban talalta el jol a vegso osztalyt.

Keplet:

```text
accuracy = helyes predikciok / osszes predikcio
```

Pelda:

- ha a modell `0.63` CT win probabilityt ad, akkor `1`-nek, vagyis CT winnek szamoljuk
- ha a valodi label is `1`, akkor ez helyes

Mikor hasznos:

- gyorsan megmutatja, mennyire jo a vegso 0/1 dontes
- jol ertheto es konnyen kommunikalhato

Mire kell figyelni:

- nem mond semmit arrol, hogy a probability mennyire jo
- egy `0.51` es egy `0.99` predikcio ugyanugy helyesnek szamit, ha a label `1`
- threshold-fuggo

Ebben a projektben:

- a legjobb futasok kb. `0.75` test accuracy korul vannak
- ez mar hasznalhato baseline
- de onmagaban nem eleg, mert a cel nem csak a nyertes osztaly, hanem a win probability is

## Precision

A precision azt meri, hogy amikor a modell CT wint prediktal, ezek kozul mennyi volt tenyleg CT win.

Keplet:

```text
precision = true positive / (true positive + false positive)
```

Jelentese:

- magas precision: ha a modell CT wint mond, akkor altalaban igaza van
- alacsony precision: sokszor mond CT wint olyan helyzetben, ahol vegul T nyer

Mikor hasznos:

- ha az a fontos, hogy a pozitiv predikciok megbizhatoak legyenek
- nalunk: ha CT win jelzesnel keves false alarmot akarunk

Mire kell figyelni:

- egy modell ugy is lehet magas precisionu, hogy nagyon ritkan mer CT wint mondani
- ezert recall nelkul nem eleg nezni

## Recall

A recall azt meri, hogy a tenyleges CT win esetekbol mennyit talalt meg a modell.

Keplet:

```text
recall = true positive / (true positive + false negative)
```

Jelentese:

- magas recall: a valodi CT win helyzetek nagy reszet felismeri
- alacsony recall: sok CT win helyzetet T winnek nez

Mikor hasznos:

- ha az a fontos, hogy keves CT win helyzet maradjon eszrevetlen

Mire kell figyelni:

- magas recall mellett lehet sok false positive is
- precisionnel egyutt kell nezni

## F1 Score

Az F1 score a precision es recall harmonikus atlaga.

Keplet:

```text
f1 = 2 * precision * recall / (precision + recall)
```

Jelentese:

- akkor magas, ha precision es recall is jo
- bunteti, ha az egyik nagyon gyenge

Mikor hasznos:

- ha egyetlen metrikaban akarjuk osszefogni a pozitiv osztaly teljesitmenyet
- kulonosen hasznos, ha az osztalyok nem teljesen kiegyensulyozottak

Mire kell figyelni:

- threshold-fuggo
- nem mer kalibraciot
- probability modellnel nem ez az egyetlen fo szempont

## Confusion Matrix

A confusion matrix megmutatja, milyen tipusu hibakat kovet el a modell.

Binary matrix:

```text
                 Pred 0      Pred 1
True 0              TN          FP
True 1              FN          TP
```

Jelentes:

- `TN`: T win volt, es a modell is T wint mondott
- `FP`: T win volt, de a modell CT wint mondott
- `FN`: CT win volt, de a modell T wint mondott
- `TP`: CT win volt, es a modell is CT wint mondott

Mikor hasznos:

- latszik, hogy milyen iranyba teved a modell
- peldaul tul sok CT wint mond-e, vagy tul ovatos-e

Ebben a projektben:

- a confusion matrix a `0.5` threshold szerinti dontest mutatja
- ha kesobb mas thresholdot akarunk hasznalni, akkor a matrix is valtozni fog

## Logloss

A logloss a probability predikciok egyik legfontosabb metrikaja.

Jelentese:

- azt bunteti, ha a modell magabiztosan rosszat mond
- minel kisebb, annal jobb

Pelda:

- ha a valodi label `1`, es a modell `0.90` CT win probabilityt ad, az jo
- ha a valodi label `1`, es a modell `0.51`-et ad, az helyes osztaly, de bizonytalan
- ha a valodi label `1`, es a modell `0.02`-t ad, az nagyon rossz, es a logloss erosen bunteti

Mikor hasznos:

- ha a probability minosege fontos
- nalunk nagyon fontos, mert CT win probabilityt akarunk ertelmezni

Mire kell figyelni:

- az accuracyvel ellentetben nem csak a threshold utani osztalyt nezi
- ezert lehet, hogy egy modell accuracyben picit jobb, de loglossban rosszabb

Ebben a projektben:

- a legjobb jelolt `xgboost_30pct_no_tick_basic_d4_n500_lr003`
- test logloss: `0.4643`
- ez jobb, mint a `d4_n600_lr003` test loglossa: `0.4652`

Ezert probability szempontbol a `d4_n500_lr003` erosebbnek tunik.

## ROC AUC

A ROC AUC azt meri, hogy a modell mennyire jol rangsorolja a pozitiv es negativ peldakat.

Jelentese:

- `0.5`: veletlenszeru rangsorolas
- `1.0`: tokeletes rangsorolas
- minel nagyobb, annal jobb

Fontos:

- nem egy konkret thresholdhoz kotott
- azt nezi, hogy a CT win esetek altalaban magasabb probabilityt kapnak-e, mint a T win esetek

Mikor hasznos:

- ha az erdekel, hogy a modell jol sorrendezi-e a helyzeteket
- win probability modellnel nagyon hasznos

Mire kell figyelni:

- nem meri, hogy a probability kalibralt-e
- egy modellnek lehet jo AUC-ja, de rossz kalibracioja

Ebben a projektben:

- a legjobb modellek test AUC-ja kb. `0.848` - `0.849`
- ez mar egesz jo rangsorolo kepesseg
- a `d4_n500_lr003` test AUC-ja `0.8490`, ami a jelenlegi futasok kozott a legerosebb

## Brier Score

A Brier score a probability predikcio negyzetes hibaja.

Keplet:

```text
brier = mean((predicted_probability - true_label)^2)
```

Jelentese:

- minel kisebb, annal jobb
- bunteti, ha a probability tavol van a valodi labeltol

Pelda:

- label `1`, predikcio `0.90`: kis hiba
- label `1`, predikcio `0.55`: kozepes hiba
- label `1`, predikcio `0.05`: nagy hiba

Mikor hasznos:

- probability minoseg meresere
- kalibracio es pontossag egyutt latszik benne

Mire kell figyelni:

- kevesbe bunteti az extrem magabiztos rossz predikciot, mint a logloss
- ezert loglossal egyutt erdemes nezni

Ebben a projektben:

- a jobb futasok test Brier score-ja kb. `0.1568` - `0.1572`
- a place/weapon kategorias futasoknal ez romlott kb. `0.16` kornyekere

## Calibration Curve

A calibration curve azt mutatja meg, hogy a modell probabilityjei mennyire felelnek meg a valos gyakorisagnak.

Pelda:

- ha a modell sok helyzetre `0.70` CT win probabilityt ad
- akkor ezekben a helyzetekben idealisan kb. `70%`-ban kellene CT winnek tortennie

Mikor jo:

- ha a pontok kozel vannak az idealis diagonalishoz
- vagyis `predicted probability ~= true frequency`

Mikor rossz:

- ha a modell tul magabiztos
- ha a modell tul ovatos
- ha bizonyos probability tartomanyokban szisztematikusan felul vagy alul becsul

Ebben a projektben:

- azert fontos, mert a modell outputjat win probabilitykent akarjuk ertelmezni
- ha a kalibracio rossz, akkor a probability nem lesz megbizhato, meg akkor sem, ha az accuracy jo

## Delta Win Probability

Ez nem klasszikus sklearn metrika, hanem modell-osszehasonlitasi otlet.

Jelentese:

- ket modell ugyanarra a sorra adott CT win probabilityjet hasonlitjuk ossze

Pelda:

```text
delta = prob_model_B - prob_model_A
```

Ha:

- `model_A`: no utility
- `model_B`: with utility

Akkor:

- pozitiv delta: utilitys modell magasabb CT win probabilityt adott
- negativ delta: utilitys modell alacsonyabb CT win probabilityt adott

Mire jo:

- nem csak azt latjuk, hogy a vegso accuracy valtozott-e
- hanem azt is, hogy mely helyzetekben mozdult el a modell hite

Utility elemzesnel kulonosen hasznos:

- meg lehet nezni, mikor aktivak a utility feature-ok
- mekkora probability delta jelenik meg fustok, mollyk, flashek, site pressure vagy recent utility damage mellett
- ott ad-e nagyobb deltat, ahol szakmailag is varjuk

Place/weapon elemzesnel is hasznos lehet:

- ha a raw kategorias modell csak random helyzetekben mozgat nagyot, az zajra utal
- ha konzisztens, ertelmezheto helyzetekben mozgat, akkor lehet benne hasznos jel

## Train-Valid/Test Gap

Ez nem kulon metrika, hanem diagnosztikai osszehasonlitas.

Pelda:

```text
auc_gap = train_auc - valid_auc
```

Jelentese:

- kis gap: jobb generalizacio
- nagy gap: tulilleszkedes gyanus

Ebben a projektben:

- `d4_n500_lr003` AUC gap: kb. `0.0462`
- `d4_n600_lr003` AUC gap: kb. `0.0535`
- `place_weapon` AUC gap: kb. `0.1023`
- `d7_n400_lr005` AUC gap: kb. `0.1399`

Ez alapjan:

- a `d4_n500_lr003` stabilabb
- a place/weapon es melyebb modellek sokkal jobban railleszkednek a train adatra

## Jelenlegi Modellfutasok Osszkep

A `modellfutasok` mappaban ezek a fo iranyok latszanak:

- `d3 / n600 / lr0.03`
- `d4 / n500-800 / lr0.03`
- `d5 / n400 / lr0.05`
- `d5 / n600 / lr0.03`
- `d6 / n400 / lr0.03`
- `d6 / n400 / lr0.05`
- `d6 / n600 / lr0.05`
- `d7 / n400 / lr0.05`
- kategorias probak: `placeonly`, `weapononly`, `place_weapon`, `withcat`

## Legjobb Jelenlegi Modell

Jelenleg a legjobb jelolt:

```text
xgboost_30pct_no_tick_basic_d4_n500_lr003
```

Fobb eredmenyek:

- valid accuracy: `0.7480`
- test accuracy: `0.7516`
- valid logloss: `0.4624`
- test logloss: `0.4643`
- valid AUC: `0.8495`
- test AUC: `0.8490`
- train-valid AUC gap: `0.0462`

Mi ezert a legjobb jelolt:

- test loglossban ez a legjobb
- test AUC-ban ez a legjobb
- train-valid gapben ez a legstabilabb
- probability modellhez ezek fontosabbak, mint az, hogy masik futas accuracyben hajszallal jobb

## Accuracy Szerinti Alternativa

Ha csak test accuracyt nezunk, akkor ez picit jobb:

```text
xgboost_30pct_no_tick_basic_d4_n600_lr003
```

Eredmenyek:

- test accuracy: `0.7526`
- test logloss: `0.4652`
- test AUC: `0.8484`

Olvasat:

- accuracyben hajszallal jobb
- de loglossban es AUC-ban picit gyengebb
- a train score magasabb, vagyis kicsit jobban illeszkedik

Ezert:

- ha csak osztalycimke kell, `d4_n600_lr003` vedheto
- ha win probability kell, `d4_n500_lr003` jobb valasztas

## Hyperparameter Tanulsag

A legerosebb minta:

- a `d4 / lr0.03` csalad mukodik a legjobban
- `n_estimators = 500` es `600` kornyeken van az eddigi jo tartomany
- `700` es `800` fa utan mar romlik a logloss
- `d6` es `d7` modellek tul erosek, jobban overfitelnek
- `lr0.05` agresszivebb, es a melyebb modellekkel kulonosen tulilleszkedik

Gyakorlati kovetkeztetes:

- most nem a nagyobb train score-t kell hajszolni
- a jobb modell az, amelyik valid/testen stabilabb
- a sekelyebb, lassabb modell jobb generalizaciot ad

## Place Feature-ok Osszefoglalasa

A raw `*_place` kategorias feature-ok:

- peldaul `T1__place`, `CT3__place`
- slotonként kulon kategorias helynev
- mapfuggo, lokalis nevezektanra epul

A place-only futas:

```text
xgboost_30pct_placeonly_d4_n600_lr003
```

Eredmenyek:

- train accuracy: `0.8534`
- valid accuracy: `0.7430`
- test accuracy: `0.7424`
- valid logloss: `0.4738`
- test logloss: `0.4785`
- valid AUC: `0.8440`
- test AUC: `0.8405`

Osszevetve a kategorianelkuli `d4_n600_lr003` futassal:

- kategorianelkul test accuracy: `0.7526`
- placeonly test accuracy: `0.7424`
- kategorianelkul test logloss: `0.4652`
- placeonly test logloss: `0.4785`
- kategorianelkul test AUC: `0.8484`
- placeonly test AUC: `0.8405`

Olvasat:

- trainen sokat javul
- valid/testen romlik
- ez erosen overfitting gyanus

Mi lehet az oka:

- a place nevek map-specifikusak
- sok kategorias splitet adnak
- konnyu megtanulni train-specifikus mintazatokat
- kozben mar vannak stabilabb pozicios feature-ok:
  - `T_macro_A/B/MID/OTHER`
  - `CT_macro_A/B/MID/OTHER`
  - `T_mean_X`, `T_mean_Y`
  - `CT_mean_X`, `CT_mean_Y`
  - `spread_xy`
  - `centroid_distance_xy`
  - `closest_enemy_dist`
  - aggregalt `T_place_*` es `CT_place_*` one-hot jellegu mezok

Kovetkeztetes:

- a raw player-slot `place` kategoria jelenlegi formaban inkabb zajos
- a vegso modellbe most nem erdemes betenni
- helyette jobb a mapfuggetlenebb, aggregalt pozicios reprezentacio

## Weapon Feature-ok Osszefoglalasa

A raw `*_primary_weapon` es `*_secondary_weapon` kategorias feature-ok:

- peldaul `T1__primary_weapon`, `CT2__secondary_weapon`
- player-slot szintu konkret fegyvernevek

A weapon-only futas:

```text
xgboost_30pct_weapononly_d4_n600_lr003
```

Eredmenyek:

- train accuracy: `0.8183`
- valid accuracy: `0.7396`
- test accuracy: `0.7477`
- valid logloss: `0.4680`
- test logloss: `0.4717`
- valid AUC: `0.8456`
- test AUC: `0.8433`

Osszevetve a kategorianelkuli `d4_n600_lr003` futassal:

- kategorianelkul test accuracy: `0.7526`
- weapononly test accuracy: `0.7477`
- kategorianelkul test logloss: `0.4652`
- weapononly test logloss: `0.4717`
- kategorianelkul test AUC: `0.8484`
- weapononly test AUC: `0.8433`

Olvasat:

- weapon kevesbe karos, mint place
- de igy sem hoz pluszt
- valid/test metrikak romlanak

Mi lehet az oka:

- a fegyver informacio nagy resze mar indirekten benne van mas feature-okben:
  - `equip_value`
  - `money`
  - `armor`
  - `helmet`
  - `scoped_count`
  - economy/equipment osszegzo feature-ok
- a konkret player-slot fegyvernev lehet tul reszletes
- a modell trainen tanul belole, de nem ad stabil pluszt valid/testen

Kovetkeztetes:

- a raw weapon kategoria jelenlegi formaban nem tunik hasznosnak
- nem annyira rossz, mint a raw place, de nem veri a kategorianelkuli alapmodellt

## Place + Weapon Egyutt

A teljes kategorias futas:

```text
xgboost_30pct_place_weapon_d4_n600_lr003
xgboost_30pct_withcat_d4_n600_lr003
```

Eredmenyek:

- train accuracy: `0.8596`
- valid accuracy: `0.7386`
- test accuracy: `0.7442`
- valid logloss: `0.4760`
- test logloss: `0.4785`
- valid AUC: `0.8414`
- test AUC: `0.8403`

Osszevetve a kategorianelkuli futassal:

- train sokkal jobb
- valid es test rosszabb
- AUC gap sokkal nagyobb

Kovetkeztetes:

- a raw place + weapon kategoria egyutt egyertelmuen tulilleszkedes iranyaba visz
- jelenlegi formaban nem javasolt a vegso modellhez

## Utility Feature-okkal Kapcsolatos Korabbi Tanulsag

A utility ablation mas mintat mutatott, mint a place/weapon.

Reduced setupban:

- kivettuk az eros `alive / hp / armor / economy / equip` blokkokat
- igy jobban latszott, hogy a utility onalloan hordoz-e jelet

Eredmeny:

- 30%-os mintan a utility javitotta a valid loglosst es AUC-t
- 50%-os mintan a utility elonye stabilabbnak tunt
- test AUC es logloss is javult

Kovetkeztetes:

- a utility feature-ok valos, de masodlagos informaciot hordoznak
- a teljes feature-keszletben ezt reszben elfedik az erosebb allapot- es economy feature-ok
- nem ez a fo informacioforras, de nem is tunik felesleges zajnak

Ez fontos kulonbseg:

- utility: valos, kisebb plusz jel
- raw place/weapon: jelenlegi formaban inkabb overfitting/zaj

## Jelenlegi Vegso Allaspont

Ha most egy modellt kellene kivalasztani:

```text
xgboost_30pct_no_tick_basic_d4_n500_lr003
```

Ezt mondanam a legjobb jelenlegi probability-modellnek.

Ha csak accuracy szamit:

```text
xgboost_30pct_no_tick_basic_d4_n600_lr003
```

De a kulonbseg kicsi, es a probability metrikak miatt a `d4_n500_lr003` szakmailag tisztabb valasztas.

## Jonak Mondhato-e A Modell

Rovid valasz:

- igen, jo baseline-nak mar mondhato
- meg nem vegleges, de mar van stabil, ertelmezheto teljesitmenye

Mi tamasztja ala:

- test accuracy kb. `0.75`
- test AUC kb. `0.849`
- valid es test eredmenyek kozel vannak
- a regularizaltabb `d4 / lr0.03` irany stabilabb, mint a melyebb modellek

Mi miatt nem vegleges:

- meg csak 30%-os mintan vannak ezek a futasok
- nincs early stopping
- nincs teljes random/grid search
- a test splitet mar tobbszor neztuk, ezert kesobb erdemes lehet friss vagy kulon holdout ellenorzes
- a kalibraciot kulon is lehetne javitani

## Javasolt Kovetkezo Lepesek

### 1. Early Stopping

Inditas:

- `max_depth = 4`
- `learning_rate = 0.03`
- nagyobb `n_estimators`, peldaul `1000`
- valid logloss alapjan early stopping

Cel:

- ne kezzel kelljen eldonteni, hogy `500`, `600` vagy `700` fa a legjobb
- a modell alljon meg ott, ahol validon mar nem javul

### 2. Finom Parameterkereses

Erdekes tartomany:

- `max_depth`: `3`, `4`, `5`
- `learning_rate`: `0.02`, `0.03`, `0.04`
- `n_estimators`: early stoppinggal nagy max ertek
- `min_child_weight`: `8`, `10`, `15`
- `subsample`: `0.65`, `0.7`, `0.8`
- `colsample_bytree`: `0.65`, `0.7`, `0.8`
- `reg_lambda`: `3`, `4`, `6`
- `reg_alpha`: `0`, `0.5`, `1`

### 3. Nagyobb Minta

A 30%-os minta jo gyors kiserletezesre.

Vegso allitashoz erdemes:

- 50%-on ujrafuttatni a legjobb par modellt
- kesobb teljes adaton is kiprobalni

### 4. Place/Weapon Ujratervezes

Raw kategoria helyett jobb lehet:

- weapon group countok:
  - rifle count
  - awp count
  - smg count
  - pistol count
  - no primary count
- oldalak kozti kulonbsegek:
  - rifle_diff
  - awp_diff
  - pistol_diff
- place helyett mapfuggetlen zona:
  - site A
  - site B
  - mid
  - spawn
  - other
- tavolsag/site pressure jellegu aggregalt feature-ok

### 5. Delta Win Probability Elemzes

Kovetkezo elemzeshez:

- vegyuk a legjobb alapmodellt
- hasonlitsuk ossze utilitys vagy mas feature-blokkos modellel
- soronkent szamoljuk a probability deltat
- nezzuk meg, mely helyzetekben mozdul nagyot

Ez segit eldonteni:

- a feature blokk csak atlagban kicsit javit
- vagy konkret, szakmailag ertelmezheto helyzetekben ad plusz informaciot

## Egy Mondatos Osszegzes

A jelenlegi eredmenyek alapjan a legjobb irany a kategorianelkuli, tick nelkuli, `max_depth = 4`, `learning_rate = 0.03`, kb. `500-600` fa koruli XGBoost modell; a utility feature-ok valos, de masodlagos jelet hordoznak, mig a raw `place` es `weapon` kategoriak jelenlegi formaban inkabb tulilleszkedest okoznak, ezert a vegso modellbe most nem tennem be oket.
