# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:21:19 2023

@author: julia
"""

#%%
# Importuje potrzebne biblioteki

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
# KATALOGI PROJEKTÓW
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"projekt 2")
KATALOG_OGOLNY = os.path.join(KATALOG_PROJEKTU,"ogolne_statystyki")
KATALOG_OSR= os.path.join(KATALOG_PROJEKTU,"okna_srednie")
KATALOG_OSTD = os.path.join(KATALOG_PROJEKTU,"okna_odchylenia")
KATALOG_WYKRESOW_OG = os.path.join(KATALOG_OGOLNY, "wykresy")
KATALOG_WYKRESOW_OSR = os.path.join(KATALOG_OSR, "wykresy")
KATALOG_WYKRESOW_OSTD = os.path.join(KATALOG_OSTD, "wykresy")
os.makedirs(KATALOG_WYKRESOW_OG, exist_ok=True)
os.makedirs(KATALOG_WYKRESOW_OSR, exist_ok=True)
os.makedirs(KATALOG_WYKRESOW_OSTD, exist_ok=True)

#%%
# DANE na których będę pracować
dane_o = pd.read_csv('wyniki.csv') #dane ogolne
dane_m = pd.read_csv('okna_sr.csv', index_col = 0) #dane okien o max sredniej
dane_s = pd.read_csv('okna_std.csv', index_col = 0) #dane okien o max std

#%%

# Usuwam kolumne z indexami, ponieważ przeszkadzałaby mi ona przy dalszej analizie
del dane_o['Unnamed: 0']

#%%
# Listy danych, aby łatwiej sie odwoływać do wszystkich tabel po kolei

dane = [dane_o, dane_m, dane_s]
wykresy = [KATALOG_WYKRESOW_OG, KATALOG_WYKRESOW_OSR, KATALOG_WYKRESOW_OSTD]

#%%
# Struktura danych

for df in dane:
    print(df.head())
    df.info()
    print('-' * 20 )
    
#%%
# Płeć jest atrybutem kategorycznym
# Wszystkie tabele dotyczą tej samej grupy pacjentów, więc poniższe wartości są takie same dla każdej tabeli
print(dane_o['Płeć'].value_counts())

#%%
# Podsumowanie atrybutów numerycznych

describe_o = dane_o.describe()
describe_m = dane_m.describe()
describe_s = dane_s.describe()

#%%
# Histogram danych
for i in range (0,len(dane)):
    dane[i].hist(figsize = (30, 30), bins = 50)
    plt.savefig(os.path.join(wykresy[i],'histogram.jpg'), dpi=300 )

#%%
# Podział danych na zbiór uczący i testujący za pomocą losowania warstwowego
# Dla kazdej tabeli po kolei

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_id_o, test_id_o in split.split(dane_o, dane_o["Wiek"]):
    train_set_o = dane_o.loc[train_id_o]
    test_set_o = dane_o.loc[test_id_o]

for train_id_m, test_id_m in split.split(dane_m, dane_m["Wiek"]):
    train_set_m = dane_m.loc[train_id_m]
    test_set_m = dane_m.loc[test_id_m]
    
for train_id_s, test_id_s in split.split(dane_s, dane_s["Wiek"]):
    train_set_s = dane_s.loc[train_id_s]
    test_set_s = dane_s.loc[test_id_s]
    
train_set = [train_set_o, train_set_m, train_set_s]
test_set = [test_set_o, test_set_m, test_set_s]

#%%
# Sprawdzenie czy losowanie warstwowe dobrze zadziałało
def porownanie(df, test, train):
    porownanie = pd.DataFrame({
        "Train" : train['Wiek'].value_counts() / len(train),
        "Dane" : df['Wiek'].value_counts() / len(df),
        "Test" : test['Wiek'].value_counts() / len(test)})
    return porownanie

for i in range(0, len(dane)):
    print(porownanie(dane[i], test_set[i], train_set[i]))
    print('-' * 20)

#%%

# MODEL 1 - DANE OGÓLNE


# ANALIZA DANYCH

# Kopia zbioru uczącego
dane1_o = train_set_o.copy()

# Szukanie korelacji
macierz_kor_o = dane1_o.corr()
print(macierz_kor_o['Wiek'].sort_values(ascending = False))

#%%

# Korelacje na wykresie
atrybuty_cor_o  = ['Wiek','p(0da)', 'p(da0)', 'p(aad)', 'p(dd)', 'p(add)']

from pandas.plotting import scatter_matrix
scatter_matrix(dane1_o[atrybuty_cor_o], figsize = (20, 20))
plt.savefig(os.path.join(wykresy[0],'korelacje.jpg'), dpi=300 )

#%%
# najbardziej obiecujący atrybut - p(0da)

plt.figure()
dane1_o.plot(x = 'p(0da)', y = 'Wiek', kind = 'scatter')

#%%
# ile atrybutów ma wysoką korelacje z innymi
print(np.sum(macierz_kor_o >= 0.85) - 1)
print(sum(np.sum(macierz_kor_o >= 0.85) - 1) / 2)

#%%
# Przykładowe atrybuty z dużą korelacją przedstawione na wykresie

plt.figure()
plt.scatter(dane1_o['25%'], dane1_o['50%'])
plt.xlabel('25%')
plt.ylabel('50%')

#%%

# PRZYGOTOWANIE DANYCH

# oddzielenie etykiet od danych
dane_be_o = train_set_o.drop('Wiek', axis =1) #dane bez etykiet
dane_etykiety_o = train_set_o['Wiek'].copy()

#%%

# podział atrybutów na kategorialne  i numeryczne

dane_cat_o = dane_be_o['Płeć']
dane_num_o = dane_be_o.drop('Płeć',  axis = 1)


#%%

# Przekształcenie danych

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder

num_atrybuty = list(dane_num_o) # takie same dla wszystkich
cat_atrybuty = ['Płeć']

transformer = ColumnTransformer([
    ('num', MinMaxScaler(), num_atrybuty),
    ('cat', OrdinalEncoder(), cat_atrybuty), 
    ])

dane_obrobione_o = transformer.fit_transform(dane_be_o)


#%%
# Przekształcenie do dataframe, aby widzieć opisane kolumny

def kolumny():
    kolumny = []
    for i in range(2, len(dane_o.columns)):
        kolumny.append(dane_o.columns[i])
    kolumny.append('Płeć')
    return kolumny

dane_obrobione_df_o = pd.DataFrame(dane_obrobione_o, columns=(kolumny()))


#%%

from sklearn.decomposition import PCA

# PCA, wyjasniające 95 % wariancji

pca95 = PCA(n_components=0.95, random_state=42)
dane95_o = pca95.fit_transform(dane_obrobione_o)

print('Liczba komponentów: ', pca95.n_components_)

print('Poszczegolne komponenty wyjasniaja % wariancji: ')
print(pca95.explained_variance_ratio_)

print('Wybrana ilosc komponentów wyjasnia {} % wariancji.'.format(round(sum(pca95.explained_variance_ratio_ * 100),2)))
print('Skumulownany % wariancji wyjasniany przez kolejne komponenty:' ,
      np.cumsum(pca95.explained_variance_ratio_ * 100))

#%%

# Przedstawienie na wykresie
plt.figure()
plt.plot(np.cumsum(pca95.explained_variance_ratio_ * 100))
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Wyjasniana wariancja [%]')
plt.title('% wariancji wyjasniany w zależnosci od liczby składowych głównych')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OG,'pca.jpg'), dpi=300 )

#%%
# Wykres wariancji

plt.figure()
plots = plt.bar(
    range(1,len(pca95.explained_variance_)+1),
    pca95.explained_variance_ratio_,
    color='pink', alpha = 0.7
    )
 
plt.plot(
    range(1,len(pca95.explained_variance_ )+1),
    np.cumsum(pca95.explained_variance_ratio_),
    c='red',
    label='Skumulowany % objasnainej wariancji')

for bar in plots:
    plt.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
plt.legend(loc='best')
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Objasniana wariancja')
plt.title('Procent wariancji objasniany przez kolejne komponenty')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OG,'pca2.jpg'), dpi=300 )

#%%
# 2 pierwsze składowe główne

pca2 = PCA(2)
projected = pca2.fit_transform(dane_obrobione_o)

plt.figure()
plt.scatter(projected[:, 0], projected[:, 1],
            c=dane_etykiety_o, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('tab10', 10))
plt.xlabel('komponent 1')
plt.ylabel('komponent 2')
plt.title('2D PCA')
plt.colorbar();
plt.savefig(os.path.join(KATALOG_WYKRESOW_OG,'pca2d.jpg'), dpi=300 )


#%%

# WYBÓR I UCZENIE MODELU


# Funkcja do porównywania błedów

def f_score(score):
    print('Wyniki: ', score)
    print('Srednia', np.mean(score))
    print('Odchylenie: ', np.std(score))
    

#%%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

sgd = SGDRegressor()

parametry = [{ 'max_iter' : [2000,3000,4500], 'eta0' : [0.01, 0.03, 0.05, 0.07, 0.1], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42], 
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_o = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_o.fit(dane95_o, dane_etykiety_o)

print(grid_search_o.best_params_)
# print(grid_search_o.cv_results_)


#%%

# wyniki dla poszczególnych parametrów

cv_wyniki_o = grid_search_o.cv_results_
for mean_score, params in zip(cv_wyniki_o["mean_test_score"], cv_wyniki_o["params"]):
    print(np.sqrt(-mean_score), params)
    
#%%
# Wyszła największa eta i najmniejsze max_iter wiec przeszukuje jeszcze raz parametry

parametry = [{ 'max_iter' : [150, 200, 250], 'eta0' : [ 0.1, 0.15, 0.2], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42],
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_o = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_o.fit(dane95_o, dane_etykiety_o)

print(grid_search_o.best_params_)

#%%
cv_wyniki_o = grid_search_o.cv_results_
for mean_score, params in zip(cv_wyniki_o["mean_test_score"], cv_wyniki_o["params"]):
    print(np.sqrt(-mean_score), params)
    
#%%
# Sprawadziann krzyżowy

sgd_o = SGDRegressor(max_iter=150, eta0 = 0.1, random_state=42 )
from sklearn.model_selection import cross_val_score
score_o = cross_val_score(sgd_o, dane95_o, dane_etykiety_o, 
                        scoring = "neg_mean_squared_error", cv = 4)
rmse_score_o = np.sqrt(-score_o)

f_score(rmse_score_o)


#%%

# Ocena za pomocą zbioru TESTOWEGO

final_model_o = grid_search_o.best_estimator_

# Oddzielam etykiety od cech
dane_be_t_o = test_set_o.drop('Wiek', axis =1) # przez t oznaczam zbior testowy
dane_etykiety_t_o = test_set_o['Wiek'].copy()

# Normalizacja
dane_obrobione_t_o = transformer.transform(dane_be_t_o)

dane95_t_o = pca95.transform(dane_obrobione_t_o)
predykcje_o = final_model_o.predict(dane95_t_o)

from sklearn.metrics import mean_squared_error
final_mse_o = mean_squared_error(dane_etykiety_t_o, predykcje_o)
final_rmse_o = np.sqrt(final_mse_o)
print(final_rmse_o)



#%%
# Przedział ufnosci
from scipy import stats
confidence = 0.95
squared_errors = (predykcje_o - dane_etykiety_t_o) ** 2
print('Przedział ufnosci: ')
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) -1, loc = squared_errors.mean(),
                       scale = stats.sem(squared_errors))))

#%%


#MODEL 2 - OKNA O NAJWIĘKSZYM SREDNIM RR



print('-' * 10 + 'MODEL 2' + '-' * 10)

# ANALIZA DANYCH

# Kopia zbioru uczącego
dane1_m = train_set_m.copy()

# Szukanie korelacji
macierz_kor_m = dane1_m.corr()
print(macierz_kor_m['Wiek'].sort_values(ascending = False))

#%%

# Korelacje na wykresie
atrybuty_cor_m  = ['Wiek','p(0d0)', 'pNN50', 'pNN20', 'RMSSD', 'std']

scatter_matrix(dane1_m[atrybuty_cor_m], figsize = (20, 20))
plt.savefig(os.path.join(wykresy[1],'korelacje.jpg'), dpi=300 )

#%%
# najbardziej obiecujący atrybut - pNN50
plt.figure()
dane1_m.plot(x = 'pNN50', y = 'Wiek', kind = 'scatter')

#%%
# ile atrybutów ma wysoką korelacje z innymi
print((np.sum(macierz_kor_m >= 0.85) - 1))
print(sum(np.sum(macierz_kor_m >= 0.85) - 1)/2)

#%%

# PRZYGOTOWANIE DANYCH

# oddzielenie etykiet od danych
dane_be_m = train_set_m.drop('Wiek', axis =1) #dane bez etykiet
dane_etykiety_m = train_set_m['Wiek'].copy()

#%%

# podział danych na kategorialne  i numeryczne

dane_cat_m = dane_be_m['Płeć']
dane_num_m = dane_be_m.drop('Płeć',  axis = 1)


#%%

# Przekształcenie danych

dane_obrobione_m = transformer.fit_transform(dane_be_m)

#%%
# Przekształcenie do dataframe, aby widzieć opisane kolumny

dane_obrobione_df_m = pd.DataFrame(dane_obrobione_m, columns=(kolumny()))

#%%

# PCA, wyjasniające 95 % wariancji

pca95 = PCA(n_components=0.95, random_state=42)
dane95_m = pca95.fit_transform(dane_obrobione_m)

print('Liczba komponentów: ', pca95.n_components_)

print('Poszczegolne komponenty wyjasniaja % wariancji: ')
print(pca95.explained_variance_ratio_)

print('Wybrana ilosc komponentów wyjasnia {} % wariancji.'.format(round(sum(pca95.explained_variance_ratio_ * 100),2)))
print('Skumulownany % wariancji wyjasniany przez kolejne komponenty:' ,
      np.cumsum(pca95.explained_variance_ratio_ * 100))

#%%

# Przedstawienie na wykresie
plt.figure()
plt.plot(np.cumsum(pca95.explained_variance_ratio_ * 100))
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Wyjasniana wariancja [%]')
plt.title('Skumulownany % wariancji wyjasniany przez kolejne komponenty:')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSR,'pca.jpg'), dpi=300 )

#%%
# Wykres wariancji

plt.figure()
plots = plt.bar(
    range(1,len(pca95.explained_variance_)+1),
    pca95.explained_variance_ratio_,
    color='pink', alpha = 0.7
    )
 
plt.plot(
    range(1,len(pca95.explained_variance_ )+1),
    np.cumsum(pca95.explained_variance_ratio_),
    c='red',
    label='Skumulowany % objasnainej wariancji')

for bar in plots:
    plt.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=8, xytext=(0, 8),
                       textcoords='offset points')
plt.legend(loc='best')
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Objasniana wariancja')
plt.title('Procent wariancji objasniany przez kolejne komponenty')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSR,'pca2.jpg'), dpi=300 )

#%%
# 2 pierwsze składowe główne

plt.figure()
pca2 = PCA(2)
projected = pca2.fit_transform(dane_obrobione_m)

plt.scatter(projected[:, 0], projected[:, 1],
            c=dane_etykiety_m, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('tab10', 10))
plt.xlabel('komponent 1')
plt.ylabel('komponent 2')
plt.title('2D PCA')
plt.colorbar();
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSR,'pca2d.jpg'), dpi=300 )
#%%

# WYBÓR I UCZENIE MODELU


#%%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

sgd = SGDRegressor()

parametry = [{ 'max_iter' : [4000,5000,5500], 'eta0' : [0.001, 0.01, 0.1], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42], 
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_m = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_m.fit(dane95_m, dane_etykiety_m)

print(grid_search_m.best_params_)


#%%

cv_wyniki_m = grid_search_m.cv_results_
for mean_score, params in zip(cv_wyniki_m["mean_test_score"], cv_wyniki_m["params"]):
    print(np.sqrt(-mean_score), params)
    
#%%

parametry = [{ 'max_iter' : [7000,8000,6500], 'eta0' : [0.001, 0.0005], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42], 
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_m = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_m.fit(dane95_m, dane_etykiety_m)

print(grid_search_m.best_params_)


#%%
# Sprawadziann krzyżowy
sgd_m = SGDRegressor(max_iter=7000, eta0 = 0.0005, random_state=42 )

score_m = cross_val_score(sgd_m, dane95_m, dane_etykiety_m, 
                        scoring = "neg_mean_squared_error", cv = 4)
rmse_score_m = np.sqrt(-score_m)

f_score(rmse_score_m)


#%%

# Ocena za pomocą zbioru TESTOWEGO

final_model_m = grid_search_m.best_estimator_

# Oddzielam etykiety od cech
dane_be_t_m = test_set_m.drop('Wiek', axis =1) #dane bez etykiet
dane_etykiety_t_m = test_set_m['Wiek'].copy()

# Normalizacja
dane_obrobione_t_m = transformer.transform(dane_be_t_m)

dane95_t_m = pca95.transform(dane_obrobione_t_m)
predykcje_m = final_model_m.predict(dane95_t_m)

final_mse_m = mean_squared_error(dane_etykiety_t_m, predykcje_m)
final_rmse_m = np.sqrt(final_mse_m)
print(final_rmse_m)

#%%
#przedziały ufnosci

squared_errors = (predykcje_m - dane_etykiety_t_m) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) -1, loc = squared_errors.mean(),
                       scale = stats.sem(squared_errors))))

#%%


#MODEL 3 - OKNA O NAJWIĘKSZYM ODCHYLENIU RR



print('-' * 10 + 'MODEL 3' + '-' * 10)

# ANALIZA DANYCH

# Kopia zbioru uczącego
dane1_s = train_set_s.copy()

# Szukanie korelacji
macierz_kor_s = dane1_s.corr()
print(macierz_kor_s['Wiek'].sort_values(ascending = False))

#%%

# Korelacje na wykresie
atrybuty_cor_s  = ['Wiek','p(0da)', 'SDNN', 'pNN50', 'max']

scatter_matrix(dane1_s[atrybuty_cor_s], figsize = (20, 20))
plt.savefig(os.path.join(wykresy[2],'korelacje.jpg'), dpi=300 )

#%%
# najbardziej obiecujący atrybut - SDNN

plt.figure()
dane1_s.plot(x = 'SDNN', y = 'Wiek', kind = 'scatter')

#%%
# ile atrybutów ma wysoką korelacje z innymi
print((np.sum(macierz_kor_s >= 0.85) - 1))
print(sum(np.sum(macierz_kor_s >= 0.85) - 1) / 2)
#%%

# PRZYGOTOWANIE DANYCH

# oddzielenie etykiet od danych
dane_be_s = train_set_s.drop('Wiek', axis =1) #dane bez etykiet
dane_etykiety_s = train_set_s['Wiek'].copy()

#%%

# podział dane na kategoryczne i numeryczne

dane_cat_s = dane_be_s['Płeć']
dane_num_s = dane_be_s.drop('Płeć',  axis = 1)


#%%

# Przekształcenie danych

dane_obrobione_s = transformer.fit_transform(dane_be_s)


#%%
# Przekształcenie do dataframe, aby widzieć opisane kolumny

dane_obrobione_df_s = pd.DataFrame(dane_obrobione_s, columns=(kolumny()))

print(dane_obrobione_df_s.head())

#%%

# PCA, wyjasniające 95 % wariancji

pca95 = PCA(n_components=0.95, random_state=42)
dane95_s = pca95.fit_transform(dane_obrobione_s)

print('Liczba komponentów: ', pca95.n_components_)

print('Poszczegolne komponenty wyjasniaja % wariancji: ')
print(pca95.explained_variance_ratio_)

print('Wybrana ilosc komponentów wyjasnia {} % wariancji.'.format(round(sum(pca95.explained_variance_ratio_ * 100),2)))
print('Skumulownany % wariancji wyjasniany przez kolejne komponenty:' ,
      np.cumsum(pca95.explained_variance_ratio_ * 100))

#%%

# Przedstawienie na wykresie
plt.figure()
plt.plot(np.cumsum(pca95.explained_variance_ratio_ * 100))
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Wyjasniana wariancja [%]')
plt.title('Skumulownany % wariancji wyjasniany przez kolejne komponenty:')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSTD,'pca.jpg'), dpi=300 )

#%%
# Wykres wariancji

plt.figure()
plots = plt.bar(
    range(1,len(pca95.explained_variance_)+1),
    pca95.explained_variance_ratio_,
    color='pink', alpha = 0.7
    )
 
plt.plot(
    range(1,len(pca95.explained_variance_ )+1),
    np.cumsum(pca95.explained_variance_ratio_),
    c='red',
    label='Skumulowany % objasnainej wariancji')

for bar in plots:
    plt.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=7, xytext=(0, 8),
                       textcoords='offset points')
plt.legend(loc='best')
plt.xlabel('Liczba składowych głównych')
plt.ylabel('Objasniana wariancja')
plt.title('Procent wariancji objasniany przez kolejne komponenty')
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSTD,'pca2.jpg'), dpi=300 )

#%%
# 2 pierwsze składowe główne

plt.figure()
pca2 = PCA(2)
projected = pca2.fit_transform(dane_obrobione_s)

plt.scatter(projected[:, 0], projected[:, 1],
            c=dane_etykiety_s, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('tab10', 10))
plt.xlabel('komponent 1')
plt.ylabel('komponent 2')
plt.title('2D PCA')
plt.colorbar();
plt.savefig(os.path.join(KATALOG_WYKRESOW_OSTD,'pca2d.jpg'), dpi=300 )
#%%

# WYBÓR I UCZENIE MODELU


#%%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

sgd = SGDRegressor()

parametry = [{ 'max_iter' : [7000,8000,8500], 'eta0' : [0.001, 0.01, 0.1], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42], 
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_s = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_s.fit(dane95_s, dane_etykiety_s)

print(grid_search_s.best_params_)
# print(grid_search_o.cv_results_)


#%%

cv_wyniki_m = grid_search_m.cv_results_
for mean_score, params in zip(cv_wyniki_m["mean_test_score"], cv_wyniki_m["params"]):
    print(np.sqrt(-mean_score), params)
    
#%%

sgd = SGDRegressor()

parametry = [{ 'max_iter' : [8000, 8500], 'eta0' : [0.001, 0.0005], 
              'tol': [1e-3],'penalty' : [None], 'random_state' :[42], 
              'learning_rate' : ['constant', 'optimal', 'invscaling']
    }]
 
grid_search_s = GridSearchCV(sgd, parametry, cv = 4, scoring= "neg_mean_squared_error")

grid_search_s.fit(dane95_s, dane_etykiety_s)

print(grid_search_s.best_params_)

#%%
# Sprawadziann krzyżowy
sgd_s = SGDRegressor(max_iter=8000, eta0 = 0.0005, random_state=42 )

score_s = cross_val_score(sgd_s, dane95_s, dane_etykiety_s, 
                        scoring = "neg_mean_squared_error", cv = 4)
rmse_score_s = np.sqrt(-score_s)

f_score(rmse_score_s)


#%%

# Ocena za pomocą zbioru TESTOWEGO

final_model_s = grid_search_s.best_estimator_

# Oddzielam etykiety od cech
dane_be_t_s = test_set_s.drop('Wiek', axis =1) #dane bez etykiet
dane_etykiety_t_s = test_set_s['Wiek'].copy()

# Normalizacja
dane_obrobione_t_s = transformer.transform(dane_be_t_s)

dane95_t_s = pca95.transform(dane_obrobione_t_s)
predykcje_s = final_model_s.predict(dane95_t_s)

final_mse_s = mean_squared_error(dane_etykiety_t_s, predykcje_s)
final_rmse_s = np.sqrt(final_mse_s)
print(final_rmse_s)


#%%
#przedziały ufnosci

squared_errors = (predykcje_s - dane_etykiety_t_s) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) -1, loc = squared_errors.mean(),
                       scale = stats.sem(squared_errors))))

#%%
# tabela porównujaca etykiety i predykcje dla najlepszego modelu

porownanie_s = pd.DataFrame(dane_etykiety_t_s)
porownanie_s['predykcje'] = predykcje_s

#%%

# Porównanie błędów w tabelach

errors = pd.DataFrame({'ogolne': [final_rmse_o], 'okna_sr' : [final_rmse_m], 'okna_std' : [final_rmse_s]})
