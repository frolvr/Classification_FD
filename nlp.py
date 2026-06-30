# Код предназначен для классификации текста с использованием метода TF-IDF и модели логистической регрессии. Мы выполняем следующие шаги:
# 
# * Загрузка данных из файлов
# * Предобработка данных
# * Разбиение описаний на предложения
# * Обучение модели TF-IDF и логистической регрессии
# * Классификация тестовых данных
# * Сохранение результатов классификации
# 
# После этого у нас есть два файла с результатами: "clean.xlsx" и "basket.xlsx", в зависимости от значения THRESHOLD.
# 
# **_clean.xlsx_** - сюда попадают примеры после классификации\
# **_basket.xlsx_** - сюда попадают примеры которые не были проклассифицированы. Эти примеры нужно изучить и снова проклассифицировать.


import pandas as pd
import numpy as np
from statistics import mean
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Глобальное определение разделителей (Предобработка)
REPLACER = {' г.': ' г', 
            ' р.': ' р',
            '?': '? ',
            '!': '! '}
DELIMIT = ('. ', '? ', '! ')


def splitter(string):
    string = string.lower().lstrip()
    for key, value in REPLACER.items():
        string = string.replace(key, value)
    
    # Компилируем регулярное выражение для повышения производительности
    regexPattern = '|'.join(map(re.escape, DELIMIT))
    compiled_regex = re.compile(regexPattern)

    # Проверка на наличие кириллических символов
    if re.search('[а-яё]', string):
        return list(filter(None, compiled_regex.split(string)))
    else:
        return np.NaN


def prediction(row, pipeline, label_encoder):
     # Объединение списка предложений в одну строку
    text = ' '.join(row['text']) if isinstance(row['text'], list) else row['text']
    
    # Преобразование текста и получение предсказаний
    response = pipeline['tfidf'].transform([text])
    encoded_result = pipeline['lr'].predict(response)
    
    result = label_encoder.inverse_transform(encoded_result)[0]
    score = np.max(pipeline['lr'].predict_proba(response), axis=1)[0]
    
    # Возвращаем результаты предсказания и вероятности
    return pd.Series({TEST_Y: result, TEST_SCORE: score})



# Порог
THRESHOLD = 0.3

# Данные
TRAIN_FILE = 'БК-База-знаний111.xlsx'
TRAIN_X = 'Вопрос'
TRAIN_Y = 'Тип-подкатегория-подподкатегория'

TEST_FILE = 'tickets-August_after_processing.xlsx'
TEST_X = 'opisanie'
TEST_Y = 'Теги'
TEST_SCORE = 'Score'


# Загрузка
logger.info('Загрузка данных ...')
train_data = pd.read_excel(TRAIN_FILE, header=0)
test_data = pd.read_excel(TEST_FILE, header=0)
logger.info('Готово')

# 0. Провести предобработку данных
logger.info('Удаление дубликатов ...')
train_data = train_data.dropna(subset=[TRAIN_Y])
test_data = test_data.dropna(subset=[TEST_X])
logger.info('Готово')

# 1. Разбить описание на предложения
logger.info('Разбиваем описание на предложения ...')
test_data['text'] = test_data[TEST_X].apply(lambda x:splitter(x))
test_data = test_data.dropna(subset=['text'])
logger.info('Готово')



# Cоздание экземпляра LabelEncoder
logger.info('Работа LabelEncoder ...')
label_encoder = LabelEncoder()

# Преобразование текстовых меток в строки
train_data[TRAIN_Y] = train_data[TRAIN_Y].apply(str)

# Преобразование текстовых меток в числовые
train_labels_encoded = label_encoder.fit_transform(train_data[TRAIN_Y])

logger.info('Готово')



# Создание pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2, max_df=0.75, max_features=20000, 
                              smooth_idf=True, use_idf=True, sublinear_tf=False, 
                              norm='l2')),
    ('lr', LogisticRegression(solver='lbfgs', multi_class='multinomial'))
])

# Обучение модели
logger.info('Обучаем модель ...')
pipeline.fit(train_data[TRAIN_X].values.astype('U'), train_labels_encoded)

logger.info('Модель обучена')



# Проверка, что test_data остается DataFrame
# logger.info(f"Тип test_data перед применением prediction: {type(test_data)}")

# Произведем классификацию
logger.info('Запуск классификации ...')
# test_data = test_data.apply(lambda row: prediction(row, pipeline), axis=1)
test_data[[TEST_Y, TEST_SCORE]] = test_data.apply(lambda row: prediction(row, pipeline, label_encoder), axis=1)
logger.info('Классификация завершена')

# Удаление вспомогательных колонок
test_data = test_data.drop(['text'], axis=1)

# Разделение данных
clean = test_data[test_data[TEST_SCORE]  >= THRESHOLD]
basket = test_data[test_data[TEST_SCORE] < THRESHOLD]

# Сохранение результатов
logger.info('Сохранение результатов ...')
clean.to_excel('clean.xlsx', index=False)
basket.to_excel('basket.xlsx', index=False)
logger.info('Результаты работы модели сохранены')



Удаление стоп-слов
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))

def splitter(string):
    # Текущая логика функции...
    # ...
    # Дополнительно: фильтрация стоп-слов из результатов
    results = [word for word in results if word not in stop_words]
    return results

Лемматизация
import spacy
nlp = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

test_data['text'] = test_data['text'].apply(lambda x: ' '.join(lemmatize(x)))


Удаление или замена чисел:
def replace_numbers(text):
    return re.sub(r'\d+', 'NUMBER', text)

test_data['text'] = test_data['text'].apply(replace_numbers)


Удаление или замена специальных символов:
def remove_special_characters(text):
    return re.sub(r'[^а-яА-Я0-9\s]', '', text)

test_data['text'] = test_data['text'].apply(remove_special_characters)





