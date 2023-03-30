# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:59:29 2022

@author: julia
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# KATALOGI PRACY 

import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"healthy_dacades")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"dane")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
KATALOG_POINCARE = os.path.join(KATALOG_WYKRESOW, "wykresy poincare")
KATALOG_HISTOGRAMY = os.path.join(KATALOG_WYKRESOW, "histogramy")
KATALOG_WIZUALIZACJI = os.path.join(KATALOG_WYKRESOW, "wizualizacje serii")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
os.makedirs(KATALOG_DANYCH, exist_ok=True)
os.makedirs(KATALOG_POINCARE, exist_ok = True)
os.makedirs(KATALOG_HISTOGRAMY, exist_ok = True)
os.makedirs(KATALOG_WIZUALIZACJI, exist_ok= True)


#%%
# ROZPAKOWANIE PLIKÓW 

import zipfile
with zipfile.ZipFile('healthy_decades.zip', 'r') as zip_ref:
    zip_ref.extractall(KATALOG_DANYCH)
    
#%%
import os

# PRZYGOTOWANIE SCIEZEK DOSTEPU DO PLIKOW
sciezki = []
for (root,dirs,files) in os.walk(KATALOG_DANYCH):
    for file in files:
        sciezki.append(os.path.join(root, file))

#%%

ilosc_pomiarow = []

# ILOSC POMIAROW W POSZCZEGOLNYCH PLIKACH        
for file in sciezki:
    count = len(open(file).readlines())
    ilosc_pomiarow.append(count)

print('--- POMIARY ---')
print('srednia liczba pomiarow:' , int(np.mean(ilosc_pomiarow)))
print('mediana: ', int(np.median(ilosc_pomiarow)))
print('najmniejsza liczba pomiarow: ',min(ilosc_pomiarow))
print('najwieksza liczba pomiarow: ', max(ilosc_pomiarow))


#%%

# SPRAWDZANIE CZY NIE MA PUSTYCH WARTOSCI
for file in sciezki:
    with open(file) as f:
        for line in f:
            if line.split()[0] == '':
                print('niekompletne')
            if line.split()[0] == '':
                print('niekompletne')
            if line.split()[0] == np.nan:
                print('niekompletne')
            if line.split()[1] == np.nan:
                print('niekompletne')
# nie ma

# SPRAWDZENIE CZY NIE MA BŁĘDNYCH WARTOSCI (UJEMNYCH)
for file in sciezki:
    with open(file) as f:
        for line in f:
            if int(line.split()[0]) <= 0:
                print('bledne dane w pliku: {} '.format(file))
                
                
#%%                    
# SPRAWDZANIE CZY SKURCZE MAJĄ KOLEJNE NUMERY 

nr = [] # wszystkie numery skurczy
numery = [] # numery skurczy z podziałem na pliki

for file in sciezki:
    numer = []
    with open(file) as f:
        for line in f:
            nr.append(int(line.split()[1]))
            numer.append(int(line.split()[1]))
        for i in range(0, len(numer)-1):
            if numer[i+1] - numer[i] != 1:
                print('blad w pliku {} , po numerze {}'.format(file[-13:], numer[i]))
    numery.append(numer)

#%%

# PRZYGOTOWANIE DANYCH

RR = []


for file in sciezki:
    with open(file) as f:
        for line in f:
            RR.append(int(line.split()[0]))

if len(RR) != len(set(nr)):
    print('numery skurczy sie powtarzają \n')
                
# STATYSTYKI DLA WSZYSTKICH DANYCH          
print('--- STATYSTYKI ---')
print('ilosc danych: ',len(RR))
print('srednie RR:' , round(np.mean(RR), 2))
print('mediana RR: ',np.median(RR))
print('najmniejsze RR: ',min(RR))
print('najwieksze RR: {} \n'.format(max(RR)))

df= pd.DataFrame(RR)
# STATYSTYKI INNY SPOSOB
print('-- STATYTSYKI INNY SPOSOB --')
print(df.describe())

# NAJCZESCIEJ WYSTEPUJACE RR
print('\nNajczesciej wystepujace wartosci: ')
print(df[0].value_counts().head())


#%%

# PRZYGOTOWANIE DANYCH
# DATAFRAME Z WSZYSTKIMI DANYMI 

def dane(skad  = sciezki, pomiary = ilosc_pomiarow):
    df2 = pd.DataFrame()
    RRp = [] #RR z podzialem na pliki

    for file in skad:
        RRf = [] #RR dla danego pliku
        with open(file) as f:
            for line in f:
                    RRf.append(int(line.split()[0]))
        RRp.append(RRf)
        if len(RRf) < max(pomiary):
            RRf1 = RRf.copy() # kopia listy RRf, aby w RRp zachować niezmienioną listę RRf, ponieważ lista jest mutable
            for i in range (max(pomiary) - len(RRf)):
                RRf1.append(np.nan)
            df2[file.split(".txt")[0][-9:]] = RRf1
        else:
            df2[file.split(".txt")[0][-9:]] = RRf

    df3 = df2.transpose()
    
    return RRp, df3, df2

df3 = dane()[1]
RRp = dane()[0]

#%%

# DODANIE KOLUMN Z WIEKIEM I PŁCIĄ

def index_plec(df):
    indexlist = df.index.tolist()
    for i in range (0,len(indexlist)):
        indexlist[i] = indexlist[i][0]
    return indexlist

def index_wiek(df):
    indexlist = df.index.tolist()
    for i in range (0,len(indexlist)):
        indexlist[i] = indexlist[i][1:3]
    return indexlist

df3.insert(0, 'Płeć', index_plec(df3))
df3.insert(1, 'Wiek', index_wiek(df3))

#%%

#STATYSTYKI KAŻDEGO PACJENTA
# print(df2.describe())
statystyki = dane()[2].describe()
wyniki = statystyki.transpose()


#%%
'''

# WYKRESY POINCARE

def poincare(nr):
    fig  = plt.figure()
    plt.scatter(df3.iloc[nr][2:-1], df3.iloc[nr][3:], lw = 0.5)
    plt.title(df3.index[nr])
    plt.xlabel(r'$RR_n [ms]$')
    plt.ylabel(r'$ RR_{n+1} [ms] $')
    fig.savefig(os.path.join(KATALOG_POINCARE,'{}.jpg'.format(df3.index[nr])), dpi=300 )
    
for i in range(0, len(df3)):
    poincare(i)    
    
#%%

# HISTOGRAM WSZYSTKICH WARTOSCI


plt.hist(RR)
plt.title('histogram akcji serca ze wszystkich danych')
plt.savefig(os.path.join(KATALOG_HISTOGRAMY,'histogram.jpg'), dpi=300 )
plt.show()

#%%

# HISTOGRAMY DLA KAŻDEGO PACJENTA

def histogram(nr):
    fig = plt.figure()
    plt.hist(RRp[nr])
    plt.title(df3.index[nr])
    plt.xlabel('RR [ms]')
    plt.savefig(os.path.join(KATALOG_HISTOGRAMY, 'histogram_{}.jpg'.format(df3.index[nr])), dpi = 300)

for i in range(0, len(df3)):
    histogram(i)



#%%

# WIZUZALIZACJE SERII DLA KAŻDEGO PACJENTA

for i in range(0, len(RRp)):
    fig = plt.figure(figsize = (20,12))
    plt.plot(numery[i], RRp[i])
    plt.title(df3.index[i])
    plt.savefig(os.path.join(KATALOG_WIZUALIZACJI , '{}.jpg'.format(df3.index[i])), dpi = 300)    
    plt.show()        
'''
#%%

# RÓŻNICE

def Roznice_f(RRp = RRp, numery = numery):
    Roznice = []
    Roznice2 = []
    for i in range(0, len(RRp)):
        roznicef = [] #roznice miedzy kolejnymi RRf w danym pliku
        roznicef2 = [] 
        for j in range (0, len(RRp[i])-1):
            if numery[i][j+1] - numery[i][j] == 1:
                roznicef.append(RRp[i][j+1] - RRp[i][j])
                roznicef2.append(RRp[i][j+1] - RRp[i][j])
            else:
                roznicef2.append('nan') 
        Roznice.append(roznicef)
        Roznice2.append(roznicef2)
    #return Roznice, Roznice2
    return Roznice, Roznice2

Roznice = Roznice_f()[0]
Roznice2 = Roznice_f()[1]
        
#%%

# ANALIZA SYGNAŁU W PRZESUWAJĄCYCH OKNACH

okna = [] # podzial na okna
oknapp = [] # okna dla poszczegolnych plikow
oknaw = [] # wyniki dla kolejnych okien
oknawp = [] # wyniki okien w plikach poszczególnych

for i in range(0,len(RRp)):
    oknap = [] # okna dla pliku
    oknawpp = []
    for j in range(0, len(RRp[i]), 100):
        if j+100 <= len(RRp[i]):
            okna.append(RRp[i][j : j + 100])
            oknap.append(RRp[i][j : j + 100])
            oknaw.append((np.mean(RRp[i][j : j+100]) , np.std(RRp[i][j : j+100])))
            oknawpp.append((np.mean(RRp[i][j : j+100]) , np.std(RRp[i][j : j+100])))
        else:
            okna.append(RRp[i][j:])
            oknap.append(RRp[i][j:])
            oknaw.append((np.mean(RRp[i][j : ]) , np.std(RRp[i][j :])))
            oknawpp.append((np.mean(RRp[i][j : ]) , np.std(RRp[i][j :])))
    oknapp.append(oknap)
    oknawp.append(oknawpp)
#%%

# WYKRES DLA SYGNAŁU W PRZESUWAJĄCYCH SIĘ OKNACH

oknadf = pd.DataFrame(oknaw)
oknadf.columns = ['mean', 'std']

fig = plt.figure(figsize = (30,12))
plt.plot(oknadf.index, oknadf['mean'])
plt.title('Analiza sygnału')
plt.ylabel('Srednia')
plt.xlabel('Numer okna')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'sygnal.jpg'), dpi=300 )

# Nie jest stacjonarny

#%%

# ANALIZA ZRÓŻNICOWANEGO SYGNAŁU W PRZESUWAJĄCYCH OKNACH

oknarw = [] #wyniki dla kolejnych okien zroznicowanego sygnalu
oknar = [] #zroznicowane okna

for i in range(0, len(Roznice)):
    for j in range(0,len(Roznice[i]), 100):
        if j+100 <= len(Roznice[i]):
            oknarw.append((np.mean(Roznice[i][j : j+100]) , np.std(Roznice[i][j : j+100])))
            oknar.append(Roznice[i][j : j+100])
        else:
            oknar.append(Roznice[i][j :])
            oknarw.append((np.mean(Roznice[i][j :]) , np.std(Roznice[i][j : ])))
            
#%%

# ANALIZA ZRÓŻNICOWANEGO SYGNAŁU WYKRES

oknardf = pd.DataFrame(oknarw)
oknardf.columns = ['mean', 'std']

fig = plt.figure(figsize = (30,12))
plt.plot(oknardf.index, oknardf['mean'])
plt.title('Analiza zróznicowanego sygnału')
plt.ylabel('Srednia')
plt.xlabel('Numer okna')
plt.savefig(os.path.join(KATALOG_WYKRESOW,'sygnal_zroznicowany.jpg'), dpi=300 )

#%%

# SDNN
# from statistics import stdev

SDNN = np.array([])

for i in range(0, len(df3)):
    SDNN = np.append(SDNN, np.std(df3.iloc[i][2:]))

wyniki['SDNN'] = SDNN
wyniki['std'] = SDNN #zamiana wyników z describe

#%%

from math import sqrt

#RMSSD

def RMSSD(r = Roznice):
    if type(r[0]) == list:
        RMSSD = np.array([])
        for i in range (0, len(r)):
            suma = 0
            for j in range(0, len(r[i])):
                suma = suma + (r[i][j])**2
            RMSSD = np.append(RMSSD, (sqrt(1/len(r[i]) * suma)))
    else:
        suma = 0
        for i in range(0, len(r)):
            suma = suma + r[i]**2
        RMSSD = sqrt(1/len(r) * suma)
    return RMSSD
    
wyniki['RMSSD'] = RMSSD()

#%%

# pNN50 

def pNN50 (r = Roznice):
    if type(r[0]) == list:
        pNN50 = np.array([])

        for i in range (0, len(r)):
            suma = 0
            for j in range(0, len(r[i])):
                if abs(r[i][j]) > 50:
                    suma += 1
            pNN50 = np.append(pNN50, suma/len(r[i]))
    else:
        suma = 0
        for i in range(0, len(r)):
            if abs(r[i]) > 50:
                suma += 1
            pNN50 = suma/len(r)
        
    return pNN50

wyniki['pNN50'] = pNN50()

#pNN20 

def pNN20(r = Roznice):
    if type(r[0]) == list:
        pNN20 = np.array([])

        for i in range (0, len(r)):
            suma = 0
            for j in range(0, len(r[i])):
                if abs(r[i][j]) > 20:
                    suma += 1
            pNN20 = np.append(pNN20, suma/len(r[i]))
    else:
        suma = 0
        for i in range(0, len(r)):
            if abs(r[i]) > 20:
                suma += 1
            pNN20 = suma/len(r)
        
    return pNN20

wyniki['pNN20'] = pNN20()
                
#%%

# SPRAWDZANIE CZY SERCE ZWALNIA CZY PRZYSPIESZA

# Zmiany serca
zmiany = []
def zmiany_serca(nr):
    zmianyf = [] #zmiany dla kazdego pliku
    ilosc_a = 0
    ilosc_d = 0
    ilosc_0 = 0
    ilosc_nan = 0
    for j in range(0, len(Roznice2[nr])):
        if Roznice2[nr][j] == 'nan':
            zmianyf.append('nan')
            ilosc_nan += 1
        elif Roznice2[nr][j] > 0:
            zmianyf.append('d')
            ilosc_d += 1
        elif Roznice2[nr][j] < 0:
            zmianyf.append('a')
            ilosc_a += 1
        else:
            zmianyf.append('0')
            ilosc_0 += 1
    return zmianyf, ilosc_a, ilosc_d, ilosc_0, ilosc_nan

p_a = []
p_d = []
p_0 = []

ilosci_nan = []

for i in range(0, len(Roznice)):
    zmiany.append(zmiany_serca(i)[0])
    ilosci_nan.append(zmiany_serca(i)[4])
    
#%%

p_a = []
p_d = []
p_0 = []

p_aa = []
p_ad = []
p_a0 = []

# P(da), P(dd), P(d0)

p_da = []
p_dd = []
p_d0 = []

# P(0a), P(0d), P(00)

p_0a = []
p_0d = []
p_00 = []

# P(aaa), P(aad), P(aa0)

p_aaa = []
p_aad = []
p_aa0 = []

# P(ada), P(add), P(ad0)

p_ada = []
p_add = []
p_ad0 = []

# P(a0a), P(a0d), P(a00)

p_a0a = []
p_a0d = []
p_a00 = []

# P(daa), P(dad), P(da0)

p_daa = []
p_dad = []
p_da0 = []

# P(dda), P(ddd), P(dd0)

p_dda = []
p_ddd = []
p_dd0 = []

# P(d0a), P(d0d), P(d00)

p_d0a = []
p_d0d = []
p_d00 = []

# P(0aa), P(0ad), P(0a0)

p_0aa = []
p_0ad = []
p_0a0 = []

# P(0da), P(0dd), P(0d0)

p_0da = []
p_0dd = []
p_0d0 = []

# P(00a), P(00d), P(000)

p_00a = []
p_00d = []
p_000 = []


def zmiany_serca2(r):
    ilosc_a = 0
    ilosc_d = 0
    ilosc_0 = 0
    
    ilosc_aa = 0
    ilosc_ad = 0
    ilosc_a0 = 0
    ilosc_da = 0
    ilosc_dd = 0
    ilosc_d0 = 0
    ilosc_0a = 0
    ilosc_0d = 0
    ilosc_00 = 0
    
    ilosc_aaa = 0
    ilosc_aad = 0
    ilosc_aa0 = 0
    
    ilosc_ada = 0
    ilosc_add = 0
    ilosc_ad0 = 0
    
    ilosc_a0a = 0
    ilosc_a0d = 0
    ilosc_a00 = 0
    
    ilosc_daa = 0
    ilosc_dad = 0
    ilosc_da0 = 0
    
    ilosc_dda = 0
    ilosc_ddd = 0
    ilosc_dd0 = 0
    
    ilosc_d0a = 0
    ilosc_d0d = 0
    ilosc_d00 = 0
    
    ilosc_0aa = 0
    ilosc_0ad = 0
    ilosc_0a0 = 0
    
    ilosc_0da = 0
    ilosc_0dd = 0
    ilosc_0d0 = 0
    
    ilosc_00a = 0
    ilosc_00d = 0
    ilosc_000 = 0
    
    for i in range(0, len(r) - 2):
        if r[i] == 'd':
            ilosc_d += 1 
            if r[i+1] == 'a':
                ilosc_da += 1
                if r[i+2] == 'a':
                    ilosc_daa += 1
                elif r[i+2] == 'd':
                    ilosc_dad += 1
                elif r[i+2] == '0':
                    ilosc_da0 += 1
            elif r[i+1] == 'd':
                ilosc_dd += 1
                if r[i+2] == 'a':
                    ilosc_dda += 1
                elif r[i+2] == 'd':
                    ilosc_ddd += 1
                elif r[i+2] == '0':
                    ilosc_dd0 += 1
            elif r[i+1] == '0':
                ilosc_d0 += 1
                if r[i+2] == 'a':
                    ilosc_d0a += 1
                elif r[i+2] == 'd':
                    ilosc_d0d += 1
                elif r[i+2] == '0':
                    ilosc_d00 += 1
        elif r[i] == 'a':
            ilosc_a += 1
            if r[i+1] == 'a':
                ilosc_aa += 1
                if r[i+2] == 'a':
                    ilosc_aaa += 1
                elif r[i+2] == 'd':
                    ilosc_aad += 1
                elif r[i+2] == '0':
                    ilosc_aa0 += 1
            elif r[i+1] == 'd':
                ilosc_ad += 1
                if r[i+2] == 'a':
                    ilosc_ada += 1
                elif r[i+2] == 'd':
                    ilosc_add += 1
                elif r[i+2] == '0':
                    ilosc_ad0 += 1
            elif r[i+1] == '0':
                ilosc_a0 += 1
                if r[i+2] == 'a':
                    ilosc_a0a += 1
                elif r[i+2] == 'd':
                    ilosc_a0d += 1
                elif r[i+2] == '0':
                    ilosc_a00 += 1
        elif r[i] == '0':
            ilosc_0 += 1
            if r[i+1] == 'a':
                ilosc_0a += 1
                if r[i+2] == 'a':
                    ilosc_0aa += 1
                elif r[i+2] == 'd':
                    ilosc_0ad += 1
                elif r[i+2] == '0':
                    ilosc_0a0 += 1
            elif r[i+1] == 'd':
                ilosc_0d += 1
                if r[i+2] == 'a':
                    ilosc_0da += 1
                elif r[i+2] == 'd':
                    ilosc_0dd += 1
                elif r[i+2] == '0':
                    ilosc_0d0 += 1
            elif r[i+1] == '0':
                ilosc_00 += 1
                if r[i+2] == 'a':
                    ilosc_00a += 1
                elif r[i+2] == 'd':
                    ilosc_00d += 1
                elif r[i+2] == '0':
                    ilosc_000 += 1
    i = len(r) - 3
    if r[i+1] == 'a':
        ilosc_a += 1
        if r[i+2] == 'a':
            ilosc_aa += 1
        elif r[i+2] == 'd':
            ilosc_ad += 1
        elif r[i+2] == '0':
            ilosc_a0 += 1
    elif r[i+1] == 'd':
        ilosc_d += 1
        if r[i+2] == 'a':
            ilosc_da += 1
        elif r[i+2] == 'd':
            ilosc_dd += 1
        elif r[i+2] == '0':
            ilosc_d0 += 1
    elif r[i+1] == '0':
        ilosc_0 += 1
        if r[i+2] == 'a':
            ilosc_0a += 1
        elif r[i+2] == 'd':
            ilosc_0d += 1
        elif r[i+2] == '0':
            ilosc_00 += 1
    
    if r[i+2] == 'a':
        ilosc_a += 1
    elif r[i+2] == 'd':
        ilosc_d += 1
    elif r[i+2] == '0':
        ilosc_0 += 1
    
    #p1 = len(r)
    #pd2 = len(r) - 1
    #pd3 = len(r) -2
    
    p2 = np.array([ilosc_aa, ilosc_ad, ilosc_a0, ilosc_da, ilosc_dd, ilosc_d0, ilosc_0a, ilosc_0d, ilosc_00,])
    #p2 = p2 / pd2
    
    p3 = np.array([ ilosc_aaa, ilosc_aad, ilosc_aa0, ilosc_ada, ilosc_add, ilosc_ad0, ilosc_a0a, ilosc_a0d, ilosc_a00, \
    ilosc_daa, ilosc_dad, ilosc_da0, ilosc_dda, ilosc_ddd, ilosc_dd0, ilosc_d0a, ilosc_d0d, ilosc_d00,\
    ilosc_0aa, ilosc_0ad, ilosc_0a0, ilosc_0da, ilosc_0dd, ilosc_0d0, ilosc_00a, ilosc_00d, ilosc_000])
    #p3 = p3/ pd3
    
    return  ilosc_a, ilosc_d, ilosc_0, p2, p3   



for i in range(0, len(zmiany)):
    # len(Roznice[i]) - 1 - liczba 2 elementowych zbiorow z kolejnych elementow listy Roznice[i]
    l1 = len(zmiany[i]) - ilosci_nan[i]
    p_a.append(zmiany_serca2(zmiany[i])[0] / l1)
    p_d.append(zmiany_serca2(zmiany[i])[1] / l1)
    p_0.append(zmiany_serca2(zmiany[i])[2] / l1)
    
    l2 = len(zmiany[i]) - 1 - 2*ilosci_nan[i]
    p_aa.append(zmiany_serca2(zmiany[i])[3][0]/ l2)
    p_ad.append(zmiany_serca2(zmiany[i])[3][1]/ l2)
    p_a0.append(zmiany_serca2(zmiany[i])[3][2] / l2)
    
    p_da.append(zmiany_serca2(zmiany[i])[3][3] / l2) 
    p_dd.append(zmiany_serca2(zmiany[i])[3][4]/ l2)
    p_d0.append(zmiany_serca2(zmiany[i])[3][5] / l2)
    
    p_0a.append(zmiany_serca2(zmiany[i])[3][6] / l2)
    p_0d.append(zmiany_serca2(zmiany[i])[3][7] / l2)
    p_00.append(zmiany_serca2(zmiany[i])[3][8] / l2)
    
    l3 = len(zmiany[i]) - 2 - 3*ilosci_nan[i]
    p_aaa.append(zmiany_serca2(zmiany[i])[4][0] / l3)
    p_aad.append(zmiany_serca2(zmiany[i])[4][1] / l3)
    p_aa0.append(zmiany_serca2(zmiany[i])[4][2] / l3)
    
    p_ada.append(zmiany_serca2(zmiany[i])[4][3] / l3)
    p_add.append(zmiany_serca2(zmiany[i])[4][4] / l3)
    p_ad0.append(zmiany_serca2(zmiany[i])[4][5] / l3)

    p_a0a.append(zmiany_serca2(zmiany[i])[4][6] / l3)
    p_a0d.append(zmiany_serca2(zmiany[i])[4][7] / l3)
    p_a00.append(zmiany_serca2(zmiany[i])[4][8] / l3)
    
    p_daa.append(zmiany_serca2(zmiany[i])[4][9] / l3)
    p_dad.append(zmiany_serca2(zmiany[i])[4][10] / l3)
    p_da0.append(zmiany_serca2(zmiany[i])[4][11] / l3)

    p_dda.append(zmiany_serca2(zmiany[i])[4][12] / l3)
    p_ddd.append(zmiany_serca2(zmiany[i])[4][13] / l3)
    p_dd0.append(zmiany_serca2(zmiany[i])[4][14] / l3)

    p_d0a.append(zmiany_serca2(zmiany[i])[4][15] / l3)
    p_d0d.append(zmiany_serca2(zmiany[i])[4][16] / l3)
    p_d00.append(zmiany_serca2(zmiany[i])[4][17] /l3)
    
    p_0aa.append(zmiany_serca2(zmiany[i])[4][18] / l3)
    p_0ad.append(zmiany_serca2(zmiany[i])[4][19] / l3)
    p_0a0.append(zmiany_serca2(zmiany[i])[4][20] / l3)

    p_0da.append(zmiany_serca2(zmiany[i])[4][21] / l3)
    p_0dd.append(zmiany_serca2(zmiany[i])[4][22] / l3)
    p_0d0.append(zmiany_serca2(zmiany[i])[4][23] / l3)

    p_00a.append(zmiany_serca2(zmiany[i])[4][24] / l3)
    p_00d.append(zmiany_serca2(zmiany[i])[4][25] / l3)
    p_000.append(zmiany_serca2(zmiany[i])[4][26] / l3)
    
    
wyniki['p(a)'] = p_a
wyniki['p(d)'] = p_d
wyniki['p(0)'] = p_0

wyniki['p(aa)'] = p_aa
wyniki['p(ad)'] = p_ad
wyniki['p(a0)'] = p_a0

wyniki['p(da)'] = p_da
wyniki['p(dd)'] = p_dd
wyniki['p(d0)'] = p_d0

wyniki['p(0a)'] = p_0a
wyniki['p(0d)'] = p_0d
wyniki['p(00)'] = p_00

wyniki['p(aaa)'] = p_aaa
wyniki['p(aad)'] = p_aad
wyniki['p(aa0)'] = p_aa0
wyniki['p(ada)'] = p_ada
wyniki['p(add)'] = p_add
wyniki['p(ad0)'] = p_ad0
wyniki['p(a0a)'] = p_a0a
wyniki['p(a0d)'] = p_a0d
wyniki['p(a00)'] = p_a00

wyniki['p(daa)'] = p_daa
wyniki['p(dad)'] = p_dad
wyniki['p(da0)'] = p_da0
wyniki['p(dda)'] = p_dda
wyniki['p(ddd)'] = p_ddd
wyniki['p(dd0)'] = p_dd0
wyniki['p(d0a)'] = p_d0a
wyniki['p(d0d)'] = p_d0d
wyniki['p(d00)'] = p_d00

wyniki['p(0aa)'] = p_0aa
wyniki['p(0ad)'] = p_0ad
wyniki['p(0a0)'] = p_0a0
wyniki['p(0da)'] = p_0da
wyniki['p(0dd)'] = p_0dd
wyniki['p(0d0)'] = p_0d0
wyniki['p(00a)'] = p_00a
wyniki['p(00d)'] = p_00d
wyniki['p(000)'] = p_000


#%%

# WZORCE KWANTOWE

# szukam max i min przyrostu

def szukaj_max():
    maximum = max(Roznice[0])
    for i in range(1, len(Roznice)):
        if max(Roznice[i]) > maximum:
            maximum  = max(Roznice[i])
    return maximum /8 

def szukaj_min():
    minimum = min(Roznice[0])
    for i in range(1, len(Roznice)):
        if min(Roznice[i]) < minimum:
            minimum  = min(Roznice[i])
    return minimum /8 

print(szukaj_max(), szukaj_min())

def wzorce(nr):
    tablica = [0] * int((szukaj_max() + abs(szukaj_min()))+1)
    for i in range(0, len(Roznice[nr])):
        j = int(Roznice[nr][i] / 8)
        tablica[j+37] = tablica[j+37] + 1
    return list(zip((np.arange(-37,38)), tablica)), tablica

wzorce_lista = []
for i in range(0, len(Roznice)):
    wzorce_lista.append(wzorce(i)[1])
    
wzorce_df = pd.DataFrame(wzorce_lista, columns=(np.arange(-37,38)), index = df3.index)



#%%

# DODANIE KOLUMNY Z PŁCIĄ I WIEKIEM

wyniki.insert(0, 'Płeć', index_plec(wyniki))
wyniki.insert(1, 'Wiek', index_wiek(wyniki))

#%%
# ZAPISANIE TABELI
wyniki.to_csv('wyniki.csv')

#%%

# DODAWANIE WYNIKOW DLA OKNA

def okno_wyniki(okna, nr, numer):
    okno_stats = []
    okno_df = pd.DataFrame(okna[nr])
    oknor1 = Roznice_f(okna, numer)[0][nr]
    oknor = Roznice_f(okna, numer)[1][nr]
    okno_zmiany = []
    stats = okno_df.describe()
    ilosc_nan =  0
    for j in range(0, len(oknor)):
        if oknor[j] == 'nan':
            okno_zmiany.append('nan')
            ilosc_nan += 1
        elif oknor[j] > 0:
            okno_zmiany.append('d')
        elif oknor[j] < 0:
            okno_zmiany.append('a')
        else:
            okno_zmiany.append('0')
    for k in range(0,8):
        okno_stats.append(stats[0][k])
    okno_stats.append(np.std(okna[nr]))
    okno_stats.append(RMSSD(oknor1))
    okno_stats.append(pNN50(oknor1))
    okno_stats.append(pNN20(oknor1))
    
    l1 = len(okno_zmiany) - ilosc_nan
    okno_stats.append(zmiany_serca2(okno_zmiany)[0] / l1)
    okno_stats.append(zmiany_serca2(okno_zmiany)[1] / l1)
    okno_stats.append(zmiany_serca2(okno_zmiany)[2] / l1)
    
    l2 = len(okno_zmiany) - 1 - 2*ilosc_nan
    for a in range(0, len(zmiany_serca2(okno_zmiany)[3])):
        okno_stats.append(zmiany_serca2(okno_zmiany)[3][a] / l2)
    
    if len(okno_zmiany) > 3:
        l3 = len(okno_zmiany) - 2 - 3*ilosc_nan
        for a in range(0, len(zmiany_serca2(okno_zmiany)[4])):
            okno_stats.append(zmiany_serca2(okno_zmiany)[4][a] / l3)
    else:
        for a in range(0, len(zmiany_serca2(okno_zmiany)[4])):
            okno_stats.append(0)
    return okno_stats

#%%

# OKNA Z MAX SR RR i MAX STD

def tworzenie_okien():
    okna_mean = []
    okna_std = []
    num_mean = []
    num_std = []
    temp = []
    for j in range(0, len(oknawp)):
        #num_plik = []
        #nums_plik = []
        temp_mean = 0
        temp_std = 0
        for i in range(0, len(oknawp[j])):
            if oknawp[j][i][0] > oknawp[j][temp_mean][0]:
                temp_mean = i
            if oknawp[j][i][1] > oknawp[j][temp_std][1]:
                temp_std = i
        okna_mean.append(oknapp[j][temp_mean])
        okna_std.append(oknapp[j][temp_std])
        temp.append(temp_mean)
        if len(okna_mean[j]) == 100: 
            num_mean.append(numery[j][temp_mean * 100 : temp_mean * 100 + 100])
        else:
            num_mean.append(numery[j][temp_mean * 100 : temp_mean * 100 + len(okna_mean[j])])
        if len(okna_std[j]) == 100:
            num_std.append(numery[j][temp_std * 100 : temp_std * 100 + 100])
        else:
            num_std.append(numery[j][temp_std * 100 : temp_std * 100 + len(okna_mean[j])])
    return okna_mean, okna_std, num_mean, num_std, temp

okna_mean, okna_std, num_mean, num_std, temp = tworzenie_okien()
    

#%%
# DODAWANIE WYNIKÓW DLA OKNA DO DATAFRAME DLA SREDNICH OKIEN
kolumny = {'count' : [], 'mean' : [], 'std':[], 'min':[], '25%':[], '50%':[], '75%':[],
       'max':[], 'SDNN':[], 'RMSSD':[], 'pNN50':[], 'pNN20':[], 'p(a)':[], 'p(d)':[], 'p(0)':[],
       'p(aa)':[], 'p(ad)':[], 'p(a0)':[], 'p(da)':[], 'p(dd)':[], 'p(d0)':[], 'p(0a)':[], 'p(0d)':[],
       'p(00)':[], 'p(aaa)':[], 'p(aad)':[], 'p(aa0)':[], 'p(ada)':[], 'p(add)':[], 'p(ad0)':[],
       'p(a0a)':[], 'p(a0d)':[], 'p(a00)':[], 'p(daa)':[], 'p(dad)':[], 'p(da0)':[], 'p(dda)':[],
       'p(ddd)':[], 'p(dd0)':[], 'p(d0a)':[], 'p(d0d)':[], 'p(d00)':[], 'p(0aa)':[], 'p(0ad)':[],
       'p(0a0)':[], 'p(0da)':[], 'p(0dd)':[], 'p(0d0)':[], 'p(00a)':[], 'p(00d)':[], 'p(000)':[]}
okna_srednie = pd.DataFrame(kolumny)
for i in range(len(okna_mean)):
    okna_srednie.loc[len(okna_srednie.index)] = okno_wyniki(okna = okna_mean, nr = i, numer = num_mean)

#%%
okna_srednie.insert(0, 'Płeć', index_plec(wyniki))
okna_srednie.insert(1, 'Wiek', index_wiek(wyniki))

okna_srednie.to_csv('okna_sr.csv')
#%%

# DODAWANIE WYNIKÓW DLA OKNA DO DATAFRAME DLA ODCHYLEN OKIEN
kolumny = {'count' : [], 'mean' : [], 'std':[], 'min':[], '25%':[], '50%':[], '75%':[],
       'max':[], 'SDNN':[], 'RMSSD':[], 'pNN50':[], 'pNN20':[], 'p(a)':[], 'p(d)':[], 'p(0)':[],
       'p(aa)':[], 'p(ad)':[], 'p(a0)':[], 'p(da)':[], 'p(dd)':[], 'p(d0)':[], 'p(0a)':[], 'p(0d)':[],
       'p(00)':[], 'p(aaa)':[], 'p(aad)':[], 'p(aa0)':[], 'p(ada)':[], 'p(add)':[], 'p(ad0)':[],
       'p(a0a)':[], 'p(a0d)':[], 'p(a00)':[], 'p(daa)':[], 'p(dad)':[], 'p(da0)':[], 'p(dda)':[],
       'p(ddd)':[], 'p(dd0)':[], 'p(d0a)':[], 'p(d0d)':[], 'p(d00)':[], 'p(0aa)':[], 'p(0ad)':[],
       'p(0a0)':[], 'p(0da)':[], 'p(0dd)':[], 'p(0d0)':[], 'p(00a)':[], 'p(00d)':[], 'p(000)':[]}
okna_odchylenia = pd.DataFrame(kolumny)
for i in range(len(okna_std)):
    okna_odchylenia.loc[len(okna_odchylenia.index)] = okno_wyniki(okna = okna_std, nr = i, numer = num_std)

okna_odchylenia.insert(0, 'Płeć', index_plec(wyniki))
okna_odchylenia.insert(1, 'Wiek', index_wiek(wyniki))

okna_odchylenia.to_csv('okna_std.csv')