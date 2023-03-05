#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# # Описания
# ## Каждая функция возвращает:
# ### records - преобразованный датасет
# ### products - уникальный список товаров
# ### transactions - транзакции

# ## Groceries_dataset.csv
# https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset?resource=download
# <br>
# Датасет содержит 38765 записей с указанием CustomerId и Даты совершения покупки

# In[28]:


def LoadFirst():
    products = []
    # Загружаем данные и преобразуем наименования столбцов для удобства
    store_data = pd.read_csv("Groceries_dataset.csv")
    store_data.rename(columns={"itemDescription":"Item", "Member_number":"CustomerId"},inplace=True)
    one_hot = pd.get_dummies(store_data['Item'])
    
    #Получаем список уникальных итемов
    products = store_data["Item"].unique()
    
    #Удаляем столбец Item за дальнейшей ненадобностью
    store_data.drop(['Item'], inplace=True, axis=1)
    store_data = store_data.join(one_hot)
    
    #Группируем по CustomerId, Date и получаем кол-во покупок итемов
    records = store_data.groupby(["CustomerId", "Date"])[products[:]].sum()
    records = records.reset_index()[products] 
    
    #Вспомогательная функция для присвоения значению имени итема
    def get_product_names(x):
        for product in products:
            if x[product] != 0:
                x[product] = product
        return x
    
    #Заменяем все не 0 значения на наименование итемов
    records = records.apply(get_product_names, axis = 1)
    
    #Формируем список транзакций состоящий из ненулевых записей
    transactions = [sub[~(sub == 0)].tolist() for sub in records.values if sub[sub != 0].tolist()]
    
    return (records, products, transactions)


# ## groceries - groceries.csv
# https://www.kaggle.com/datasets/irfanasrullah/groceries
# <br>
# Набор данных содержит 9835 транзакций клиентов, покупающих продукты. Данные содержат 169 уникальных элементов.

# In[8]:


def LoadSecond():
    products = []
    # Загружаем данные
    store_data = pd.read_csv("groceries - groceries.csv")
    
    #Получаем список уникальных итемов
    uniqueProducts = set()
    for row in store_data.values:
        for column in row:
            if(isinstance(column, str)):
                uniqueProducts.add(column)
    products = list(uniqueProducts)
    
    # Удаляем ненужный столбец
    store_data.drop(['Item(s)'], inplace=True, axis=1)
    
    #Вместо NaN ставим 0
    for sub in store_data.values:
        sub[pd.isna(sub)] = 0
        
    #Поскольку датасет уже в удобном нам формате, то никаких доп. манипуляций не нужно
    #Формируем список транзакций состоящий из ненулевых записей
    transactions = [sub[~(sub == 0)].tolist() for sub in store_data.values if sub[sub != 0].tolist()]
    
    return (store_data, products, transactions)


# ## march_trs_data_clean.csv
# https://www.kaggle.com/datasets/christopherbarran/realworldretailtransactions
# <br>
# Датасет состоит из 560493 записей. Из-за большого количество используем только 50000. В результате датасет содержит 5282 уникальных транзакций и 5849 уникальных итемов

# In[2]:


def LoadThird():
    products = []
    # Загружаем данные
    store_data = pd.read_csv("march_trs_data_clean.csv")
    store_data.rename(columns={"ItemDescription":"Item"},inplace=True)
    store_data = store_data.drop_duplicates()
    store_data = store_data[:50000]
    
    one_hot = pd.get_dummies(store_data['Item'])
    
    #Получаем список уникальных итемов
    products = store_data["Item"].unique()
    
    #Удаляем столбец Item за дальнейшей ненадобностью
    store_data.drop(['Item'], inplace=True, axis=1)
    store_data = store_data.join(one_hot)

    #Группируем по TransactionID и получаем кол-во покупок итемов
    records = store_data.groupby(["TransactionID"])[products[:]].sum()
    records = records.reset_index()[products] 

    #Вспомогательная функция для присвоения значению имени итема
    def get_product_names(x):
        for product in products:
            if x[product] != 0:
                x[product] = product
        return x

    #Заменяем все не 0 значения на наименование итемов
    records = records.apply(get_product_names, axis = 1)

    #Формируем список транзакций состоящий из ненулевых записей
    transactions = [sub[~(sub == 0)].tolist() for sub in records.values if sub[sub != 0].tolist()]
    
    return (records, products, transactions)

