
## Funkcja XOR

Piotr Drabik

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

Analiza sprawności wyuczonej sieci:

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

## [Link do repozytorium z kodem źródłowym](https://github.com/piotr233/neural-nets-cpp)
https://github.com/piotr233/neural-nets-cpp