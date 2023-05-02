# test_OK
# Данная версия является первой. Обновленная версия (с учетом фидбека) содержится в другой ветке
- В файле ```main.py``` содержится основное решение задания. Именно при его запуске выдаются рекомендации.

- В файле ```normalise_data.csv``` содержится кросс-таблица, строки которой - названия исполнителей, а столбцы - id пользователя. На их пересечении стоит значение ```scrobbles```.
- В файле ```pearson_corr_data.csv``` содержится таблица, строки и столбцы которой - названия исполнителей. В данной таблице хранится корреляция Пирсона.
- В файле ```cosine_data.csv``` содержится таблица, строки и столбцы которой - названия исполнителей. В данной таблице хранится косинусная мера близости.
- В файле ```indices_pearson.csv``` содержится таблица, строки которой - названия исполнителей и в ней хранятся индексы 5 ближайших (по корреляции) соседей.
- В файле ```indices_cosine.csv``` содержится таблица, строки которой - названия исполнителей и в ней хранятся индексы 5 ближайших (по косинусной мере) соседей.
- В файле ```КазаковаАС_ОК.ipynb``` хранится работа с данными и получение всех ```.csv``` файлов, а также ход моих мыслей, вопросы и идеи, возникшие в ходе выполнения данного задания.

# Необходимые библиотеки
```Pandas```

# Запуск
1. Для запуска программы необходимо, чтобы файлы ```normalise_data.csv```, ```pearson_corr_data.csv```, ```cosine_data.csv```, ```indices_pearson.csv```, ```indices_cosine.csv``` были в одной директории с ```main.py```.

2. В терминале вводим
```
python main.py [arg]
```
  где ``` [arg] ``` отвечает за тип запускаемого алгоритма. Всего их четыре вида:
  
  - ```alg1``` выводит рекомендации на основе ближайших соседей, использующих корреляцию. Этот алгоритм выдаёт быстрые ответы.
  
  - ```alg2``` выводит рекомендации на основе ближайших соседей, использующих косинусную меру близости. Этот алгоритм также выдаёт быстрые ответы.
  
  - ```alg3``` выводит рекомендации на основе матрицы схожести, посчитанной с помощью косинусной меры близости. Этот алгоритм работает долго из-за того, что необходимо считать   матрицу 17492x17492. При чем этот алгоритм дает (почти) одинаковые ответы с alg2.
  
  - ```alg4``` выводит рекомендации на основе матрицы схожести, посчитанной с помощью корреляции Пирсона. Этот алгоритм работает долго, но немного быстрее alg3  (также из-за     того, что необходимо считать матрицу 17492x17492).
  
3. В поле  ввода необходимо ввести название исполнителя и нажать ```enter```.

   На вход принимаются только значения, которые были в изначальной таблице, причем регистр, пробелы и спецю символы должны быть записаны также, как в таблице. Почему так объяснено в ```.ipynb``` файле
   
   Для выхода из программы необходимо ввести  ```stop```
   
# Пример выполнения

![screenshot-1](https://github.com/NastyAnanasty/test_OK/blob/main/img/screenshot.png)

![screenshot-2](https://github.com/NastyAnanasty/test_OK/blob/main/img/screenshot-2.png)

