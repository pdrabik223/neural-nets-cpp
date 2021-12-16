# Neural Net

Piotr Drabik

## Opis narzędzia

"Neural Net" to narzędzie umożliwiające tworzenie prymitywnych gęstych sieci neuronowych. Oparte w pełni na operacjach
macierzowych, w aktualnym stanie pozwala na uczenie wzmocnione sieci o dowolnej ilości warstw ukrytych, wielkości
wektora wejściowego czy wyjściowego. Do dyspozycji użytkownika oddawana jest także kontrola nad funkcją liczenia błędu (
Cost Function), jak i funkcją aktywacji neurona (Activation function).

W aktualnym stanie Narzędzie jest w stanie skutecznie tworzyć i uczyć sieci gęste.

## XOR

### Wstęp

Funkcja xor przyjmuje dwie wartości boolowskie i zwraca efekt ich xorowania w postaci doubla.

```c++
double LinearFunction(int x, int y) { return x ^ y; }
```

Funkcja przyjmuje i zwraca wartości 0 lub 1.

Sieć zaprojektowana do aproksymacji danej funkcji:

```c++
 NeuralNet neural_net(2, {4}, 1);
```

Warstwa wejściowa ma wielkość 2 neuronów, posiada jedną warstwę ukrytą o wielkości 4 neuronów. Na wyjściu zwracana jest
jedna wartość.

Na każdą z warstw sieci nakładana jest funkcja Aktywacji Relu.

Przyjęty learning rate = 0.1 ze spadkiem 0.04999 co 500 powtórzeń cyklu uczenia.

### Opis wyników uczenia sieci

Wykres błędu sieci na przedziale pierwszych 600 iteracji wacha się w zakresie 0.1 do 0.6, po czym drastycznie spada, by
przed 1000 powtórzeniem się cyklu uczenia ustabilizować się na 0;
![xor_learning](xor_learning.png)
Wykres błędu od iteracji pętli uczenia się sieci

### Analiza sprawności wyuczonej sieci:

Tabela przedstawia losowo wygenerowanych 10 par x,y wynik funkcji xor dla nich i wynik funkcji feed forward

| test no | x value  | y value  | correct answer | net estimation |
| :-----: | :------: | :------: | :------------: | :------------: |
|    0    | 1.000000 | 0.000000 |    1.000000    |    1.000000    |
|    1    | 0.000000 | 1.000000 |    1.000000    |    1.000000    |
|    2    | 1.000000 | 1.000000 |    0.000000    |    0.000000    |
|    3    | 1.000000 | 1.000000 |    0.000000    |    0.000000    |
|    4    | 1.000000 | 1.000000 |    0.000000    |    0.000000    |
|    5    | 0.000000 | 0.000000 |    0.000000    |    0.000000    |
|    6    | 0.000000 | 0.000000 |    0.000000    |    0.000000    |
|    7    | 0.000000 | 1.000000 |    1.000000    |    1.000000    |
|    8    | 0.000000 | 0.000000 |    0.000000    |    0.000000    |
|    9    | 1.000000 | 1.000000 |    0.000000    |    0.000000    |

### Wnioski

Zastosowanie warstwy ukrytej o wysokości 4 neuronów, odpowiadający czterem kombinacjom wartości {0,1} pozwala na idealne
wyuczenie sieci funkcji xor.

## MNIST dataset

Mnist dataset to zbiór 70 000 obrazów ręczenie malowanych cyfr, każdy obraz ma rozmiar 28x28 pixeli, na każdy pixel
składa sie 8 bitowa wartość, która odpowiada odcieniowi szarości tego pixel'a.

Do każdego z obrazów przypisana jest wartość w zakresie <0;9>

By przystosować dane do procesu uczenia, każdy pixel zdjęcia został znormalizowany i pixele zdjęcia zostały rzędami
ułożone w wektor o długości 728 pixeli.

Label obrazka zamieniono na wektor o długości 10, wypełniony zerami. Na pozycji odpowiadającej opisowi obrazka
wstawiono wartość jeden.

Przykład wektora label, dla wartości 4: [0,0,0,0,1,0,0,0,0,0].

### Opis sieci

Użyta do wyuczenia zbioru sieć ma kształt:

| trait / Layer | Input Linear | Hidden Linear 1 | Hidden Linear 2 | Output Linear |
| :-----------: | :---------: | :------------: | :------------: | :----------: |
|     Size      |     728     |       16       |       16       |      10      |
| Cost Function |    Relu     |      Relu      |      Relu      |   Sigmoid    |

Do liczenia błędu użyto potęgowej wartości błędu, (x-oczekiwana)^2.

### Opis uczenia sieci

W procesie uczenia wykorzystywany był zbiór 60 000 obrazków, by ostatnie 10 000 odłożyć do testów. Algorytm przez zbiór
danych uczących przeszedł 10 razy, w losowej kolejności.

Learning rate przez cały okres uczenia sieci pozostawał stały równy 0.1.

Wykres uczenia się, czyli błąd sieci względem powtórzeń pętli uczenia.

Wartość błędu to skumulowana różnica kwadratów odpowiadających sobie elementów: wyjściowego wektora sieci i wektora
oczekiwanej odpowiedzi.

![mnist running error](C:\Users\piotr\Documents\neural-nets-cpp\mnist_running_error.png)

### Analiza sprawności sieci

Do testów sprawności sieci użyto 10 000 elementowego zestawu testowego, na przestrzeni wszystkich przypadków testowych
średnia błędu wynosi: 0.243251;

Wartości label i wynik sieci porównane w tabeli.

|            |      0       |      1       |      2       |    3     |      4       |      5       |    6     |      7       |    8     |      9       |
| :--------: | :----------: | :----------: | :----------: | :------: | :----------: | :----------: | :------: | :----------: | :------: | :----------: |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 | **1.000000** | 0.000000 |   0.000000   |
| net aprox: |   0.010057   |   0.075192   |   0.082242   | 0.033942 |   0.023430   |   0.112871   | 0.022212 | **0.944743** | 0.026148 |   0.008690   |
|   label:   |   0.000000   |   0.000000   | **1.000000** | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.055281   |   0.000231   | **0.687127** | 0.005613 |   0.000070   |   0.000935   | 0.046551 |   0.004082   | 0.000838 |   0.006410   |
|   label:   |   0.000000   | **1.000000** |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.001652   | **0.947228** |   0.057317   | 0.023301 |   0.034559   |   0.010866   | 0.010149 |   0.059811   | 0.012042 |   0.011531   |
|   label:   | **1.000000** |   0.000000   |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: | **0.986538** |   0.000000   |   0.013422   | 0.000022 |   0.000000   |   0.048063   | 0.016593 |   0.000011   | 0.000027 |   0.003025   |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 | **1.000000** |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.004064   |   0.008222   |   0.084756   | 0.024816 | **0.926747** |   0.009792   | 0.026202 |   0.059874   | 0.007852 |   0.012809   |
|   label:   |   0.000000   | **1.000000** |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.000634   | **0.979875** |   0.020418   | 0.015358 |   0.028423   |   0.004370   | 0.002802 |   0.024951   | 0.006420 |   0.009274   |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 | **1.000000** |   0.000000   | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.003553   |   0.011207   |   0.013360   | 0.039137 | **0.892206** |   0.018799   | 0.001636 |   0.040165   | 0.061069 |   0.046969   |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 | **1.000000** |
| net aprox: |   0.024922   |   0.016885   |   0.019036   | 0.057473 |   0.130775   |   0.030521   | 0.001082 |   0.091700   | 0.028889 | **0.874089** |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 |   0.000000   | **1.000000** | 0.000000 |   0.000000   | 0.000000 |   0.000000   |
| net aprox: |   0.046831   |   0.000261   |   0.006778   | 0.000196 |   0.001085   | **0.109677** | 0.904271 |   0.001142   | 0.006428 |   0.000246   |
|   label:   |   0.000000   |   0.000000   |   0.000000   | 0.000000 |   0.000000   |   0.000000   | 0.000000 |   0.000000   | 0.000000 | **1.000000** |
| net aprox: |   0.031273   |   0.007716   |   0.015832   | 0.028301 |   0.132347   |   0.019835   | 0.000779 |   0.052765   | 0.011965 | **0.942439** |

Należy szczególną uwagę zwrócić na różnicę pomiędzy zwracanymi przez sieć wartościami.

Każda ze zwracanych wartości może być interpretowana w następujący sposób: "na ile procent sieć jest pewna, że analizowany obraz jest daną cyfrą". 

Dla pierwszej pary, label, net approximation "sieć jest pewna" że na zdjęciu widnieje zero na 0.01%;

Więcej wniosków można wynieść z analizy podawanego obrazu.  
Label oznacza poprawną wartość, 
net estimation: zinterpretowaną wartość zwróconą przez sieć,
output: wartość zwróconą przez sieć,
error: błąd konkretnego przykładu.
![images.ong](examples/MNIST-try/images.png)

### Wnioski
Sieć skutecznie nauczyła się wykrywać cyfry na zdjęciach zbioru danych mnist.
Uczenie sieci można było skrócić, wykres uczenia sieci sugeruje, że po 10 000 iteracji pętli uczenia się nie nastąpił, znaczy skok jakości uczonego modelu.