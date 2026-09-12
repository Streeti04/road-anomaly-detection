# Part A: Understanding the Problem – Ausarbeitung (A.1 bis A.5)

---

## A.1 — Arbeiten ohne Labels

### (a) Warum nicht einfach einen überwachten Schadensklassifikator benutzen?

* **Anderer Datensatz, andere Bilder:** Öffentliche Datensätze wie RDD2020 sehen komplett anders aus als unsere Cyclomedia-Aufnahmen (8192 × 6144 px, fester Kamerawinkel, man sieht immer die Motorhaube, eigene Beleuchtungssituation). Ein Modell, das auf RDD2020 trainiert wurde, erkennt auf unseren Bildern ohne großes Fine-Tuning fast nichts wieder.
* **Wir haben gar keine Labels:** In unseren 10.000 Bildern gibt es keine einzige Annotation, ob eine Straße beschädigt ist oder nicht. Alles von Hand zu labeln wäre extrem aufwendig und teuer – und genau das soll das Projekt ja automatisieren.
* **Glücklicherweise ist fast alles „normal“:** Über 90–95 % der Straßen sind intakt. Das ist ideal für einen unüberwachten Ansatz: Das Modell lernt, wie intakter Asphalt aussieht, und alles, was davon abweicht, wird als Anomalie markiert. Dafür brauchen wir keine gelabelten Schadensbilder.

### (b) Was kann beim Rekonstruktionsfehler schiefgehen (und was tun wir dagegen)?

Zwei Beispiele, wo ein hoher Rekonstruktionsfehler entsteht, ohne dass tatsächlich ein Straßenschaden da ist:

1. **Harte Schatten (von Bäumen oder Gebäuden):**
   * *Problem:* Extreme Helligkeitskontraste und harte Schattenkanten – sowas kann der Autoencoder schlecht rekonstruieren und der Fehler schießt hoch.
   * *Gegenmaßnahme in Part B (Preprocessing):* Helligkeit und Kontrast normalisieren (z. B. mit CLAHE) oder in einen Farbraum wechseln, wo Schatten weniger stören (z. B. Helligkeitskanal in LAB/HSV glätten).
2. **Fahrbahnmarkierungen, Kanaldeckel, Gras am Straßenrand:**
   * *Problem:* Weiße Markierungen, metallische Deckel oder Grasbüschel am Maskenrand erzeugen alle einen hohen Rekonstruktionsfehler – sind aber keine Schäden.
   * *Gegenmaßnahme in Part C (Segmentation):* Die Segmentierungsmaske leicht erosieren (morphologische Erosion), damit Randbereiche mit Vegetation/Kanten rausfallen, und die Segmentierung so prompten, dass Markierungen und Schächte ausgeschlossen werden.

---

## A.2 — Bildentstehung und Frequenzgehalt

### (a) Lochkameramodell, Fluchtpunkt-Perspektive und Patch-Sampling

* **Projektionsgeometrie (Pinhole-Modell):**  
  Beim Lochkameramodell gilt für die Bildkoordinate $y = f \cdot \frac{Y}{Z}$, wobei $Z$ die Tiefe (Abstand zur Kamera) ist. Das heißt: Je weiter etwas weg ist, desto weniger Pixel bekommt es – die Auflösung (Pixel pro Meter Straße) skaliert mit $\frac{1}{Z}$.
* **Warum feste Patch-Größen ein Problem sind:**  
  Ein fester $128 \times 128$-Patch zeigt unten im Bild (Nähe, kleines $Z$) nur ein paar Dezimeter Belag – mit allen feinen Texturdetails. Oben im Bild (Ferne, großes $Z$) zeigt derselbe Patch aber gleich mehrere Dutzend Meter Straße – Details verschmieren und die perspektivische Verzerrung dominiert.
* **Was das für Part D (Sampling) bedeutet:**  
  * Entweder das Sampling auf den **Nahbereich** beschränken (unteres Drittel bzw. untere Hälfte der Straßenmaske – dort ist $Z$ annähernd konstant),
  * oder in Part B ein **Inverse Perspective Mapping (IPM / Bird's-Eye-View)** machen, bzw. die Patch-Größe dynamisch je nach Bildhöhe $y$ skalieren.

### (b) 2D-Fouriermagnitude: Asphalt vs. Vegetation – welche Frequenzen unterscheiden sie?

* **Sauberer Asphalt:**  
  * Das Spektrum ist **weitgehend kreissymmetrisch (isotrop)** – die Energie verteilt sich gleichmäßig in alle Richtungen.
  * Sie fällt von niedrigen zu hohen Frequenzen gleichmäßig ab. Das ist einfach das zufällige Kornmuster der Asphaltoberfläche.
* **Vegetation (Blätter/Gras):**  
  * Deutlich **mehr Energie in mittleren und hohen Frequenzen**.
  * Das Spektrum ist **stark richtungsabhängig (anisotrop)** – man sieht deutliche Spitzen senkrecht zu den dominanten Kanten (Stängel, Blattränder).
* **Welche Frequenzen unterscheiden am besten?**  
  Vor allem die **mittleren bis hohen Frequenzen** und wie sich ihre Energie **über die Richtungen verteilt**. So lassen sich strukturierte Kanten (Vegetation, Risse, Markierungen) vom diffusen, isotropen Asphaltkorn unterscheiden.

---

## A.3 — Straße segmentieren ohne Labels

### (a) Auswahlkriterium und typische Fehlerquellen

* **Unser Kriterium (mit SAM / promptable Segmentation):**  
  Wir setzen mit Point-Prompts an: ein positiver Punkt unten in der Bildmitte (auf der Fahrbahn, direkt vor der Motorhaube) plus negative Prompts an den Rändern (Himmel, Motorhaube). Damit nutzen wir die vortrainierten Repräsentationen von SAM.
* **Warum ein starres Kriterium scheitert:**  
  Bei 10.000 Bildern ändert sich alles Mögliche: Position der Fahrspur, Straßenkrümmung, Abbiegen, unterschiedliche Beläge (Asphalt, Beton, Pflaster).
* **Kritisches Beispiel:**  
  **Starke Kurven oder Kreuzungen:** Steht das Auto in einer Kurve, landet der feste Punkt in der Bildmitte plötzlich neben der Fahrbahn – auf dem unbefestigten Rand, einer Verkehrsinsel oder der Gegenfahrbahn.

### (b) Masken-Auswahlheuristik und Failure Case

* **Heuristik:**  
  *Wir nehmen die größte zusammenhängende Maske, die den unteren Ankerpunkt enthält (Fahrbahnmitte vor der Motorhaube) und sich nach oben trapezförmig verjüngt.*
* **Failure Case:**  
  **Weicher Übergang zu Schotter-/Feldwegen ohne Bordstein:**  
  Wenn der Fahrbahnrand ohne Farb- oder Texturkontrast in Erdboden oder Schotter übergeht, kann die Segmentierung Straße und Feld nicht mehr trennen – es entsteht eine einzige riesige Maske. Unsere Heuristik wählt dann fälschlicherweise das Ganze als „Straße“ aus.

---

## A.4 — Generative Modelle als Anomalie-Detektoren

### (a) Die formale Annahme – und was der Rekonstruktionsfehler wirklich sagt

* **Die Annahme:**  
  Sei $p_{\text{normal}}(x)$ die Verteilung von normalen, intakten Straßen-Patches. Der Autoencoder $f_\theta$ wird darauf trainiert, $\mathbb{E}_{x \sim p_{\text{normal}}}[\|x - f_\theta(x)\|^2]$ zu minimieren. Die Kernannahme ist:
  $$\mathcal{L}_{\text{rec}}(x) = \|x - f_\theta(x)\|^2 \text{ ist klein} \iff x \sim p_{\text{normal}}$$
* **Was ein hoher Rekonstruktionsfehler bedeutet:**  
  Der Patch ist visuell „out of distribution“, also untypisch im Vergleich zu den Trainingsdaten.
* **Was er NICHT bedeutet:**  
  Er sagt nichts darüber aus, *was* der Patch ist. Es ist keine Schadensklassifikation (Schlagloch vs. Markierung vs. Ölfleck vs. Blatt vs. Kanaldeckel – der AE unterscheidet die nicht) und keine Aussage über den Schweregrad des Schadens.
* **Unterschied zu einem überwachten Klassifikator:**  
  Der Klassifikator liefert $P(y = \text{Schaden} \mid x)$, weil er gelernt hat, wie Schäden aussehen. Der Autoencoder liefert nur ein Maß dafür, „das ist ungewöhnlich“ – ohne zu wissen, warum.

### (b) Trade-Offs: Adversarial (AnoGAN) vs. Autoencoder / VAE

1. **Trainingstabilität:**  
   GANs sind berüchtigt für Mode Collapse und instabiles Minimax-Training. Ein normaler AE/VAE optimiert eine simple Rekonstruktionsloss (MSE/L1) – das konvergiert stabil und reproduzierbar.
2. **Inferenzgeschwindigkeit:**  
   AnoGAN muss für jeden einzelnen Test-Patch eine iterative Optimierung im latenten Raum machen ($z^* = \arg\min \mathcal{L}(G(z), x)$). Bei hunderten Patches pro Bild × 10.000 Bildern ist das viel zu langsam. Ein AE/VAE macht stattdessen nur einen einzigen Forward-Pass.
* **Unsere Entscheidung:**  
  Wir nehmen einen **Convolutional Autoencoder (bzw. VAE)**. In einem 8-Wochen-Projekt ohne gelabeltes Validierungsset ist das die pragmatische Wahl: stabiles Training, schnelle Inferenz auf vielen Patches, und der Rekonstruktionsfehler ist direkt interpretierbar.

---

## A.5 — Modell-Annahmen

| # | Annahme | Auswirkung auf die Detektion | Verzerrungsrichtung |
|---|---|---|---|
| **1** | **Intakte Straßen dominieren:** Der ungelabelte Trainingspool besteht fast nur aus intaktem Asphalt. | Wenn bestimmte Schäden (z. B. Längsrisse oder Bitumen-Flickstellen) sehr häufig im Trainingsset vorkommen, lernt der Autoencoder sie als „normal“ und erkennt sie nicht mehr als Anomalie. | **Under-estimation** (zustandsbewertung zu positiv, False Negatives bei häufigen Kleinschäden). |
| **2** | **Visuelle Anomalie = Straßenschaden:** Jeder hohe Rekonstruktionsfehler wird als sanierungsrelevanter Schaden interpretiert. | Harmlose Objekte (weiße Pfeile, Bitumennähte, Kanaldeckel, Ölflecken, Blätter) erzeugen ebenfalls hohe Rekonstruktionsfehler. | **Over-estimation** (Bewertung zu pessimistisch, viele False Positives). |
| **3** | **Feste Patch-Größe ($N \times N$ px), keine Skalenanpassung:** Die perspektivische Verkürzung wird ignoriert. | Nahe Patches zeigen feinste Mikrotexturen/Risse im Millimeterbereich; ferne Patches decken zig Quadratmeter ab – selbst große Schäden verschmieren dort zu wenigen Pixeln und werden gut rekonstruiert. | **Distanzabhängiger Bias:** überempfindlich im Nahbereich, blind für Schäden im Fernbereich. |
| **4** | **Die Straßenmaske ist perfekt:** Die Segmentierung trennt Asphalt sauber von Randstreifen, Gras und Motorhaube. | Unscharfe Maskenränder schneiden Gras, Bordsteine oder die Motorhaubenkante mit ein – das erzeugt extreme Rekonstruktionsfehler genau an den Rändern. | **Over-estimation** an Maskenrändern und unkuratierten Straßenabschnitten (Randartefakt-Bias). |
