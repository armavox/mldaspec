
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/web/677/8e1/337/6778e1337c3d4b159d7e99df94227cb2.jpg"/>
# ## Специализация "Машинное обучение и анализ данных"
# <center>Автор материала: программист-исследователь Mail.Ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ [Юрий Кашницкий](https://yorko.github.io/)

# # <center> Capstone проект №1 <br> Идентификация пользователей по посещенным веб-страницам
# <img src='http://i.istockimg.com/file_thumbview_approve/21546327/5/stock-illustration-21546327-identification-de-l-utilisateur.jpg'>
# 
# # <center>Неделя 4.  Сравнение алгоритмов классификации
# 
# Теперь мы наконец подойдем к обучению моделей классификации, сравним на кросс-валидации несколько алгоритмов, разберемся, какие параметры длины сессии (*session_length* и *window_size*) лучше использовать. Также для выбранного алгоритма построим кривые валидации (как качество классификации зависит от одного из гиперпараметров алгоритма) и кривые обучения (как качество классификации зависит от объема выборки).
# 
# **План 4 недели:**
# - Часть 1. Сравнение нескольких алгоритмов на сессиях из 10 сайтов
# - Часть 2. Выбор параметров – длины сессии и ширины окна
# - Часть 3. Идентификация  конкретного пользователя и кривые обучения
#  
# 
# 
# **В этой части проекта Вам могут быть полезны видеозаписи следующих лекций курса "Обучение на размеченных данных":**
#    - [Линейная классификация](https://www.coursera.org/learn/supervised-learning/lecture/jqLcO/linieinaia-klassifikatsiia)
#    - [Сравнение алгоритмов и выбор гиперпараметров](https://www.coursera.org/learn/supervised-learning/lecture/aF79U/sravnieniie-alghoritmov-i-vybor-ghipierparamietrov)
#    - [Кросс-валидация. Sklearn.cross_validation](https://www.coursera.org/learn/supervised-learning/lecture/XbHEk/kross-validatsiia-sklearn-cross-validation)
#    - [Линейные модели. Sklearn.linear_model. Классификация](https://www.coursera.org/learn/supervised-learning/lecture/EBg9t/linieinyie-modieli-sklearn-linear-model-klassifikatsiia)
#    - и многие другие
# 

# ### Задание
# 1. Заполните код в этой тетрадке 
# 2. Если вы проходите специализацию Яндеса и МФТИ, пошлите файл с ответами в соответствующем Programming Assignment. <br> Если вы проходите курс ODS, выберите ответы в [веб-форме](https://docs.google.com/forms/d/12VB7kmzDoSVzSpQNaJp0tR-2t8K8PynQopP3dypf7i4).  

# In[1]:


# pip install watermark
get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,pandas,matplotlib,statsmodels,sklearn -g')


# In[3]:


from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
from time import time
import itertools
import os
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pickle
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


# In[4]:


# Поменяйте на свой путь к данным
PATH_TO_DATA = 'data'


# In[5]:


def write_answer_to_file(value, filename):
    with open(filename, 'w') as fout:
            fout.writelines([str(item) + ' ' for item in value])


# ## Часть 1. Сравнение нескольких алгоритмов на сессиях из 10 сайтов

# **Загрузим сериализованные ранее объекты *X_sparse_10users* и *y_10users*, соответствующие обучающей выборке для 10 пользователей.**

# In[6]:


with open(os.path.join(PATH_TO_DATA, 'X_sparse_10users.pkl'), 'rb') as X_sparse_10users_pkl:
    X_sparse_10users = pickle.load(X_sparse_10users_pkl)
with open(os.path.join(PATH_TO_DATA, 'y_10users.pkl'), 'rb') as y_10users_pkl:
    y_10users = pickle.load(y_10users_pkl)


# **Здесь более 14 тысяч сессий и почти 5 тысяч уникальных посещенных сайтов.**

# In[7]:


X_sparse_10users.shape


# **Разобьем выборку на 2 части. На одной будем проводить кросс-валидацию, на второй – оценивать модель, обученную после кросс-валидации.**

# In[8]:


X_train, X_valid, y_train, y_valid = train_test_split(X_sparse_10users, y_10users, 
                                                      test_size=0.3, 
                                                      random_state=17, stratify=y_10users)


# **Зададим заранее тип кросс-валидации: 3-кратная, с перемешиванием, параметр random_state=17 – для воспроизводимости.**

# In[9]:


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)


# **Вспомогательная функция для отрисовки кривых валидации после запуска GridSearchCV (или RandomizedCV).**

# In[10]:


def plot_validation_curves(param_values, grid_cv_results_):
    train_mu, train_std = grid_cv_results_['mean_train_score'], grid_cv_results_['std_train_score']
    valid_mu, valid_std = grid_cv_results_['mean_test_score'], grid_cv_results_['std_test_score']
    train_line = plt.plot(param_values, train_mu, '-', label='train', color='dodgerblue')
    valid_line = plt.plot(param_values, valid_mu, '-', label='test', color='darkorange')
    plt.fill_between(param_values, train_mu - train_std, train_mu + train_std, edgecolor='none',
                     facecolor=train_line[0].get_color(), alpha=0.2)
    plt.fill_between(param_values, valid_mu - valid_std, valid_mu + valid_std, edgecolor='none',
                     facecolor=valid_line[0].get_color(), alpha=0.2)
    plt.legend()


# ### K-Nearest Neighbors

# **1. Обучите `KNeighborsClassifier` со 100 ближайшими соседями (остальные параметры оставьте по умолчанию, только `n_jobs`=-1 для распараллеливания) и посмотрите на долю правильных ответов на 3-кратной кросс-валидации (ради воспроизводимости используйте для этого объект `StratifiedKFold` `skf`) по выборке `(X_train, y_train)` и отдельно на выборке `(X_valid, y_valid)`.**

# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


knn = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
knn.fit(X_train, y_train)


# **<font color='red'>Вопрос 1. </font> Посчитайте доли правильных ответов для KNeighborsClassifier на кросс-валидации и отложенной выборке. Округлите каждое до 3 знаков после запятой и введите через пробел.**

# In[13]:


cv_score = cross_val_score(knn, X_train, y_train, cv=skf, n_jobs=-1)
acc_score = accuracy_score(y_valid, knn.predict(X_valid))


# In[14]:


print(f'CV score: {cv_score.mean():.3f}')
print(f'Valid score: {acc_score:.3f}')


# In[15]:


write_answer_to_file([round(cv_score.mean(), 3),
                      round(acc_score, 3)],
                     'answer4_1.txt')
get_ipython().system('cat answer4_1.txt')


# ### Random Forest

# **2. Обучите случайный лес (`RandomForestClassifier`) из 100 деревьев (для воспроизводимости `random_state`=17). Посмотрите на OOB-оценку (для этого надо сразу установить `oob_score`=True) и на долю правильных ответов на выборке `(X_valid, y_valid)`. Для распараллеливания задайте `n_jobs`=-1.**

# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


forest = RandomForestClassifier(n_estimators=100, random_state=17, oob_score=True, n_jobs=-1)
forest.fit(X_train, y_train)


# In[18]:


acc_score = accuracy_score(y_valid, forest.predict(X_valid))


# In[19]:


print(f'Out-of-Bag score: {forest.oob_score_:.3f}')
print(f'Valid score: {acc_score:.3f}')


# **<font color='red'>Вопрос 2. </font> Посчитайте доли правильных ответов для `RandomForestClassifier` при Out-of-Bag оценке и на отложенной выборке. Округлите каждое до 3 знаков после запятой и введите через пробел.**

# In[20]:


write_answer_to_file([round(forest.oob_score_, 3),
                     round(accuracy_score(y_valid, forest.predict(X_valid)), 3)],
                     'answer4_2.txt')
get_ipython().system('cat answer4_2.txt')


# ### Logistic Regression

# **3. Обучите логистическую регрессию (`LogisticRegression`) с параметром `C` по умолчанию и `random_state`=17 (для воспроизводимости). Посмотрите на долю правильных ответов на кросс-валидации (используйте объект `skf`, созданный ранее) и на выборке `(X_valid, y_valid)`. Для распараллеливания задайте `n_jobs=-1`.**

# In[21]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# In[22]:


logit = LogisticRegression(random_state=17, n_jobs=-1)
logit.fit(X_train, y_train)


# In[23]:


cv_score = cross_val_score(logit, X_train, y_train, cv=skf).mean()
acc_score = accuracy_score(y_valid, logit.predict(X_valid))
print(f'CV score: {cv_score:.3f}')
print(f'Valid score: {acc_score:.3f}')


# **Почитайте документацию к [LogisticRegressionCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html). Логистическая регрессия хорошо изучена, и для нее существуют алгоритмы быстрого подбора параметра регуляризации `C` (быстрее, чем с `GridSearchCV`).**
# 
# **С помощью `LogisticRegressionCV` подберите параметр `C` для `LogisticRegression` сначала в широком диапазоне: 10 значений от 1e-4 до 1e2, используйте `logspace` из `NumPy`. Укажите у `LogisticRegressionCV` параметры `multi_class`='multinomial' и `random_state`=17. Для кросс-валидации используйте объект `skf`, созданный ранее. Для распараллеливания задайте `n_jobs=-1`.**
# 
# **Нарисуйте кривые валидации по параметру `C`.**

# In[24]:


logit_c_values1 = np.logspace(-4, 2, 10)

logit_grid_searcher1 = LogisticRegressionCV(logit_c_values1, cv=skf, 
                                            multi_class='multinomial', 
                                            random_state=17, verbose=1, n_jobs=-1)
logit_grid_searcher1.fit(X_train, y_train)


# Средние значения доли правильных ответов на кросс-валидации по каждому из 10 параметров `C`.

# In[25]:


logit_grid_searcher1.C_.mean()


# In[26]:


C_scores = []
for key, value in logit_grid_searcher1.scores_.items():
    C_scores.append(logit_grid_searcher1.scores_[key].mean(axis=0))
logit_mean_cv_scores1 = np.asarray(C_scores).mean(axis=0)
print(logit_mean_cv_scores1)


# Выведите лучшее значение доли правильных ответов на кросс-валидации и соответствующее значение `C`.

# In[27]:


logit_mean_cv_scores1.max(), logit_grid_searcher1.Cs_[logit_mean_cv_scores1.argmax()]


# Нарисуйте график зависимости доли правильных ответов на кросс-валидации от `C`.

# In[28]:


plt.figure(figsize=(12,4))
plt.plot(np.log10(logit_c_values1), logit_mean_cv_scores1)
plt.grid(True, alpha=.4, axis='x')
plt.xlabel('log10(C)')
plt.ylabel('Accuracy score')
for x, y in zip(np.log10(logit_c_values1), logit_mean_cv_scores1):
    plt.annotate(str(round(y, 3)), (x, y), (x-.5, y+.03), arrowprops={'arrowstyle': '->'})


# **Теперь то же самое, только значения параметра `C` перебирайте в диапазоне `np.linspace`(0.1, 7, 20). Опять нарисуйте кривые валидации, определите максимальное значение доли правильных ответов на кросс-валидации.**

# In[29]:


logit_c_values2 = np.linspace(0.1, 7, 20)

logit_grid_searcher2 = LogisticRegressionCV(logit_c_values2, cv=skf, 
                                            multi_class='multinomial', 
                                            random_state=17, verbose=1, n_jobs=-1)
logit_grid_searcher2.fit(X_train, y_train)


# In[30]:


logit_grid_searcher2.C_.mean()


# Средние значения доли правильных ответов на кросс-валидации по каждому из 10 параметров `C`.

# In[31]:


C_scores = []
for key, value in logit_grid_searcher2.scores_.items():
    C_scores.append(logit_grid_searcher2.scores_[key].mean(axis=0))
logit_mean_cv_scores2 = np.asarray(C_scores).mean(axis=0)


# Выведите лучшее значение доли правильных ответов на кросс-валидации и соответствующее значение `C`.

# In[32]:


logit_best_C = logit_grid_searcher2.Cs_[logit_mean_cv_scores2.argmax()]
logit_mean_cv_scores2.max(), logit_best_C


# Нарисуйте график зависимости доли правильных ответов на кросс-валидации от `C`.

# In[33]:


plt.figure(figsize=(12,4))
plt.plot(logit_c_values2, logit_mean_cv_scores2)
plt.xlabel('C')
plt.ylabel('Accuracy score')
plt.grid(True, alpha=.4, axis='x')
for x, y in zip(logit_c_values2, logit_mean_cv_scores2):
    plt.annotate(str(round(y, 3)), (x, y), (x+.2, y-.004), arrowprops={'arrowstyle': '->'});


# Выведите долю правильных ответов на выборке `(X_valid, y_valid)` для логистической регрессии с лучшим найденным значением `C`.

# **<font color='red'>Вопрос 3. </font>Посчитайте доли правильных ответов для `logit_grid_searcher2` на кросс-валидации для лучшего значения параметра `C` и на отложенной выборке. Округлите каждое до 3 знаков после запятой и выведите через пробел.**

# In[35]:


get_ipython().run_cell_magic('time', '', "logit_grid_searcher2.Cs = [logit_best_C]\nlogit_cv_acc = accuracy_score(y_valid, logit_grid_searcher2.predict(X_valid))\ncv_score = cross_val_score(logit_grid_searcher2, X_train, y_train, cv=skf, n_jobs=-1)\n\nprint(f'CV score: {cv_score.mean():.3f}')\nprint(f'Valid score: {logit_cv_acc:.3f}')")


# In[36]:


write_answer_to_file([round(cv_score.mean(), 3),
                     round(logit_cv_acc, 3)],
                    'answer4_3.txt')
get_ipython().system('cat answer4_3.txt')


# ### Linear SVC

# **4. Обучите линейный SVM (`LinearSVC`) с параметром `C`=1 и `random_state`=17 (для воспроизводимости). Посмотрите на долю правильных ответов на кросс-валидации (используйте объект `skf`, созданный ранее) и на выборке `(X_valid, y_valid)`.**

# In[37]:


from sklearn.svm import LinearSVC


# In[38]:


svm = LinearSVC(random_state=17).fit(X_train, y_train)
acc_score = accuracy_score(y_valid, svm.predict(X_valid))
cv_score = cross_val_score(svm, X_train, y_train, cv=skf, n_jobs=-1)
print(f'CV score: {cv_score.mean():.3f}')
print(f'Valid score: {acc_score:.3f}')


# **С помощью `GridSearchCV` подберите параметр `C` для SVM сначала в широком диапазоне: 10 значений от 1e-4 до 1e4, используйте `linspace` из NumPy. Нарисуйте кривые валидации.**

# In[39]:


get_ipython().run_cell_magic('time', '', "svm_params1 = {'C': np.linspace(1e-4, 1e4, 10)}\n\nsvm_grid_searcher1 = GridSearchCV(svm, svm_params1, return_train_score=True, n_jobs=-1)\nsvm_grid_searcher1.fit(X_train, y_train)")


# Выведите лучшее значение доли правильных ответов на кросс-валидации и соответствующее значение `C`.

# In[40]:


svm_grid_searcher1.best_score_, svm_grid_searcher1.best_params_


# Нарисуйте график зависимости доли правильных ответов на кросс-валидации от `C`.

# In[41]:


plot_validation_curves(svm_params1['C'], svm_grid_searcher1.cv_results_)
plt.xlabel('C')
plt.ylabel('Accuracy score');


# **Но мы помним, что с параметром регуляризации по умолчанию (С=1) на кросс-валидации доля правильных ответов выше. Это тот случай (не редкий), когда можно ошибиться и перебирать параметры не в том диапазоне (причина в том, что мы взяли равномерную сетку на большом интервале и упустили действительно хороший интервал значений `C`). Здесь намного осмысленней подбирать `C` в районе 1, к тому же, так модель быстрее обучается, чем при больших `C`. **
# 
# **С помощью `GridSearchCV` подберите параметр `C` для SVM в диапазоне (1e-3, 1), 30 значений, используйте `linspace` из NumPy. Нарисуйте кривые валидации.**

# In[42]:


get_ipython().run_cell_magic('time', '', "svm_params2 = {'C': np.linspace(1e-3, 1, 30)}\n\nsvm_grid_searcher2 = GridSearchCV(svm, svm_params2, cv=skf, return_train_score=True, n_jobs=-1)\nsvm_grid_searcher2.fit(X_train, y_train)")


# Выведите лучшее значение доли правильных ответов на кросс-валидации и соответствующее значение `C`.

# In[43]:


svm_grid_searcher2.best_score_, svm_grid_searcher2.best_params_


# Нарисуйте график зависимости доли правильных ответов на кросс-валидации от С.

# In[44]:


plot_validation_curves(svm_params2['C'], svm_grid_searcher2.cv_results_)
plt.xlabel('C')
plt.ylabel('Accuracy score');


# Выведите долю правильных ответов на выборке `(X_valid, y_valid)` для `LinearSVC` с лучшим найденным значением `C`.

# In[45]:


svm_best_C = svm_grid_searcher2.best_params_['C']


# In[46]:


svm = LinearSVC(C=svm_best_C, random_state=1).fit(X_train, y_train)
print(f'{(accuracy_score(y_valid, svm.predict(X_valid))):.3f}')


# **<font color='red'>Вопрос 4. </font> Посчитайте доли правильных ответов для `svm_grid_searcher2` на кросс-валидации для лучшего значения параметра `C` и на отложенной выборке. Округлите каждое до 3 знаков после запятой и выведите через пробел.**

# In[47]:


svm_cv_acc = accuracy_score(y_valid, svm_grid_searcher2.best_estimator_.predict(X_valid))
cv_score = cross_val_score(svm_grid_searcher2.best_estimator_, X_train, y_train, cv=skf, n_jobs=-1)

print(f'CV score: {cv_score.mean():.3f}')
print(f'Valid score: {svm_cv_acc:.3f}')


# In[48]:


write_answer_to_file([round(cv_score.mean(), 3),
                      round(svm_cv_acc, 3)],
                     'answer4_4.txt')
get_ipython().system('cat answer4_4.txt')


# ## Часть 2. Выбор параметров – длины сессии и ширины окна

# **Возьмем `LinearSVC`, показавший лучшее качество на кросс-валидации в 1 части, и проверим его работу еще на 8 выборках для 10 пользователей (с разными сочетаниями параметров *session_length* и *window_size*). Поскольку тут уже вычислений побольше, мы не будем каждый раз заново подбирать параметр регуляризации `C`.**
# 
# **Определите функцию `model_assessment`, ее документация описана ниже. Обратите внимание на все детали. Например, на то, что разбиение  выборки с `train_test_split` должно быть стратифицированным. Не теряйте нигде `random_state`.**

# In[49]:


def model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv, random_state=17, test_size=0.3):
    """Estimates CV-accuracy for (1 - test_size) share of (X_sparse, y) 
    loaded from path_to_X_pickle and path_to_y_pickle and holdout accuracy for (test_size) share of (X_sparse, y).
    The split is made with stratified train_test_split with params random_state and test_size.
    
    :param estimator – Scikit-learn estimator (classifier or regressor)
    :param path_to_X_pickle – path to pickled sparse X (instances and their features)
    :param path_to_y_pickle – path to pickled y (responses)
    :param cv – cross-validation as in cross_val_score (use StratifiedKFold here)
    :param random_state –  for train_test_split
    :param test_size –  for train_test_split
    
    :returns mean CV-accuracy for (X_train, y_train) and accuracy for (X_valid, y_valid) where (X_train, y_train)
    and (X_valid, y_valid) are (1 - test_size) and (testsize) shares of (X_sparse, y).
    """
    
    with open(path_to_X_pickle, 'rb') as X_pickle:
        X = pickle.load(X_pickle)
    with open(path_to_y_pickle, 'rb') as y_pickle:
        y = pickle.load(y_pickle)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size,
                                                         random_state=random_state, stratify=y)
#     print(X_train.shape, y_train.shape)
    cv_score = cross_val_score(estimator, X_train, y_train, cv=cv, n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    acc_score = accuracy_score(y_valid, estimator.predict(X_valid))
    
    return cv_score.mean(), acc_score


# **Убедитесь, что функция работает.**

# In[50]:


get_ipython().run_cell_magic('time', '', "cv, val = model_assessment(svm_grid_searcher2.best_estimator_, \n                           os.path.join(PATH_TO_DATA, 'X_sparse_10users.pkl'),\n                           os.path.join(PATH_TO_DATA, 'y_10users.pkl'), cv=skf, test_size=0.3,\n                           random_state=17)\nprint(f'CV score: {cv:.3f}')\nprint(f'Valid score: {val:.3f}')")


# **Примените функцию *model_assessment* для лучшего алгоритма из предыдущей части (а именно, `svm_grid_searcher2.best_estimator_`) и 9 выборок вида с разными сочетаниями параметров *session_length* и *window_size* для 10 пользователей. Выведите в цикле параметры *session_length* и *window_size*, а также результат вывода функции *model_assessment*. 
# Удобно сделать так, чтоб *model_assessment* возвращала 3-им элементом время, за которое она выполнилась. На моем ноуте этот участок кода выполнился за 20 секунд. Но со 150 пользователями каждая итерация занимает уже несколько минут.**

# Здесь для удобства стоит создать копии ранее созданных pickle-файлов X_sparse_10users.pkl, X_sparse_150users.pkl, y_10users.pkl и y_150users.pkl, добавив к их названиям s10_w10, что означает длину сессии 10 и ширину окна 10. 

# In[128]:


get_ipython().system('cp $PATH_TO_DATA/X_sparse_10users.pkl $PATH_TO_DATA/X_sparse_10users_s10_w10.pkl ')
get_ipython().system('cp $PATH_TO_DATA/X_sparse_150users.pkl $PATH_TO_DATA/X_sparse_150users_s10_w10.pkl ')
get_ipython().system('cp $PATH_TO_DATA/y_10users.pkl $PATH_TO_DATA/y_10users_s10_w10.pkl ')
get_ipython().system('cp $PATH_TO_DATA/y_150users.pkl $PATH_TO_DATA/y_150users_s10_w10.pkl ')


# In[51]:


get_ipython().run_cell_magic('time', '', "estimator = svm_grid_searcher2.best_estimator_\nusers = 10\nfor window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):\n    if window_size <= session_length:\n        path_to_X_pkl = os.path.join(PATH_TO_DATA, f'X_sparse_{users}users_s{session_length}_w{window_size}.pkl')\n        path_to_y_pkl = os.path.join(PATH_TO_DATA, f'y_{users}users_s{session_length}_w{window_size}.pkl')\n        print(path_to_X_pkl[-11:-4], model_assessment(estimator, path_to_X_pkl, path_to_y_pkl, cv=skf))")


# **<font color='red'>Вопрос 5. </font> Посчитайте доли правильных ответов для `LinearSVC` с настроенным параметром `C` и выборки `X_sparse_10users_s15_w5`. Укажите доли правильных ответов на кросс-валидации и на отложенной выборке. Округлите каждое до 3 знаков после запятой и выведите через пробел.**

# In[52]:


path_to_X_sparse_10users_s15_w5 = os.path.join(PATH_TO_DATA, f'X_sparse_10users_s15_w5.pkl')
path_to_y_10users_s15_w5 = os.path.join(PATH_TO_DATA, f'y_10users_s15_w5.pkl')

best_window_scores = model_assessment(estimator, 
                                      path_to_X_sparse_10users_s15_w5, 
                                      path_to_y_10users_s15_w5, 
                                      cv=skf)


# In[53]:


print(f'CV scores: {best_window_scores[0]:.3f}')
print(f'Valid scores: {best_window_scores[1]:.3f}')


# In[54]:


write_answer_to_file([round(best_window_scores[0], 3),
                      round(best_window_scores[1], 3)],
                      'answer4_5.txt')
get_ipython().system('cat answer4_5.txt')


# **Прокомментируйте полученные результаты. Сравните для 150 пользователей доли правильных ответов на кросс-валидации и оставленной выборке для сочетаний параметров (*session_length, window_size*): (5,5), (7,7) и (10,10). На среднем ноуте это может занять до часа – запаситесь терпением, это Data Science :) **
# 
# **Сделайте вывод о том, как качество классификации зависит от длины сессии и ширины окна.**

# In[55]:


get_ipython().run_cell_magic('time', '', "estimator = svm_grid_searcher2.best_estimator_\nusers=150\nfor window_size, session_length in [(5,5), (7,7), (10,10)]:\n    path_to_X_pkl = os.path.join(PATH_TO_DATA, f'X_sparse_{users}users_s{session_length}_w{window_size}.pkl')\n    path_to_y_pkl = os.path.join(PATH_TO_DATA, f'y_{users}users_s{session_length}_w{window_size}.pkl')\n    print(path_to_X_pkl[-11:-4], model_assessment(estimator, path_to_X_pkl, path_to_y_pkl, cv=skf))")


# **<font color='red'>Вопрос 6. </font> Посчитайте доли правильных ответов для `LinearSVC` с настроенным параметром `C` и выборки `X_sparse_150users`. Укажите доли правильных ответов на кросс-валидации и на отложенной выборке. Округлите каждое до 3 знаков после запятой и выведите через пробел.**

# In[56]:


get_ipython().run_cell_magic('time', '', "path_to_X_pkl = os.path.join(PATH_TO_DATA, f'X_sparse_150users.pkl')\npath_to_y_pkl = os.path.join(PATH_TO_DATA, f'y_150users.pkl')\n\nbest_150users_scores = model_assessment(estimator, \n                                        path_to_X_pkl, \n                                        path_to_y_pkl, \n                                        cv=skf)")


# In[57]:


print(f'CV score: {best_150users_scores[0]:.3f}')
print(f'Validation score: {best_150users_scores[1]:.3f}')


# In[58]:


write_answer_to_file([round(best_150users_scores[0], 3),
                      round(best_150users_scores[1], 3)],
                      'answer4_6.txt')
get_ipython().system('cat answer4_6.txt')


# ## Часть 3. Идентификация  конкретного пользователя и кривые обучения

# **Поскольку может разочаровать, что многоклассовая доля правильных ответов на выборке из 150 пользовалей невелика, порадуемся тому, что конкретного пользователя можно идентифицировать достаточно хорошо. **

# **Загрузим сериализованные ранее объекты *X_sparse_150users* и *y_150users*, соответствующие обучающей выборке для 150 пользователей с параметрами (*session_length, window_size*) = (10,10). Так же точно разобьем их на 70% и 30%.**

# In[59]:


with open(os.path.join(PATH_TO_DATA, 'X_sparse_150users.pkl'), 'rb') as X_sparse_150users_pkl:
     X_sparse_150users = pickle.load(X_sparse_150users_pkl)
with open(os.path.join(PATH_TO_DATA, 'y_150users.pkl'), 'rb') as y_150users_pkl:
    y_150users = pickle.load(y_150users_pkl)


# In[60]:


X_train_150, X_valid_150, y_train_150, y_valid_150 = train_test_split(
    X_sparse_150users, y_150users, test_size=0.3, random_state=17, stratify=y_150users)


# **Обучите `LogisticRegressionCV` для одного значения параметра `C` (лучшего на кросс-валидации в 1 части, используйте точное значение, не на глаз). Теперь будем решать 150 задач "Один-против-Всех", поэтому укажите аргумент `multi_class`='ovr'. Как всегда, где возможно, указывайте `n_jobs=-1` и `random_state`=17.**

# In[61]:


get_ipython().run_cell_magic('time', '', "logit_cv_150users = LogisticRegressionCV(Cs=[logit_best_C], multi_class='ovr', cv=skf, \n                                         random_state=17, n_jobs=-1)\nlogit_cv_150users.fit(X_train_150, y_train_150)")


# **Посмотрите на средние доли правильных ответов на кросс-валидации в задаче идентификации каждого пользователя по отдельности.**

# In[62]:


cv_scores_by_user = {}
for user_id in logit_cv_150users.scores_:
    cv_scores_by_user[user_id] = logit_cv_150users.scores_[user_id].mean()
    print('User {}, CV score: {:.4f}'.format(user_id, logit_cv_150users.scores_[user_id].mean()))


# **Результаты кажутся впечатляющими, но возможно, мы забываем про дисбаланс классов, и высокую долю правильных ответов можно получить константным прогнозом. Посчитайте для каждого пользователя разницу между долей правильных ответов на кросс-валидации (только что посчитанную с помощью `LogisticRegressionCV`) и долей меток в *y_train_150*, отличных от ID 
#  этого пользователя (именно такую долю правильных ответов можно получить, если классификатор всегда "говорит", что это не пользователь с номером $i$ в задаче классификации $i$-vs-All).**

# In[63]:


class_distr = np.bincount(y_train_150.astype('int'))
user_share = {}
acc_diff_vs_constant = {}

for user_id in np.unique(y_train_150):
    user_share[user_id] = class_distr[user_id] / (class_distr.sum())
    acc_diff_vs_constant[user_id] = cv_scores_by_user[user_id] - (1 - user_share[user_id])


# In[64]:


num_better_than_default = (np.array(list(acc_diff_vs_constant.values())) > 0).sum()
num_better_than_default


# **<font color='red'>Вопрос 7. </font> Посчитайте долю пользователей, для которых логистическая регрессия на кросс-валидации дает прогноз лучше константного. Округлите до 3 знаков после запятой.**

# In[65]:


.907 * 150


# In[66]:


better_than_default_share = num_better_than_default / 150


# In[67]:


write_answer_to_file([round(better_than_default_share, 3)],
                    'answer4_7.txt')
get_ipython().system('cat answer4_7.txt')


# **Дальше будем строить кривые обучения для конкретного пользователя, допустим, для 128-го. Составьте новый бинарный вектор на основе *y_150users*, его значения будут 1 или 0 в зависимости от того, равен ли ID-шник пользователя 128.**

# In[71]:


from sklearn.model_selection import learning_curve

def plot_learning_curve(val_train, val_test, train_sizes, 
                        xlabel='Training Set Size', ylabel='score'):
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                         facecolor=lines[0].get_color(), alpha=0.2)
    plot_with_err(train_sizes, val_train, label='train')
    plot_with_err(train_sizes, val_test, label='valid')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(loc='lower right');


# **Посчитайте доли правильных ответов на кросс-валидации в задаче классификации "user128-vs-All" в зависимости от размера выборки. Не помешает посмотреть встроенную документацию для *learning_curve*.**

# In[72]:


'''Почему-то без train-test split не работает lerning curve'''
y_binary_128 = np.array(y_150users == 128, dtype='int')

X_train_150, _, y_train_150, _ = train_test_split(
    X_sparse_150users, y_binary_128, test_size=0.01, random_state=17, stratify=y_binary_128)


# In[73]:


get_ipython().run_cell_magic('time', '', 'train_sizes = np.linspace(0.25, 1, 20)\nn_train, val_train, val_test = learning_curve(logit_cv_150users, X_train_150, y_train_150, \n                                              train_sizes=train_sizes, cv=skf, n_jobs=-1,\n                                              random_state=17)')


# In[74]:


plt.figure(figsize=(12,4))
plt.grid(True, alpha=.4)
plot_learning_curve(val_train, val_test, n_train, 
                    xlabel='train_size', ylabel='accuracy')


# **Сделайте выводы о том, помогут ли алгоритму новые размеченные данные при той же постановке задачи.**

# ## Пути улучшения
# - конечно, можно проверить еще кучу алгоритмов, например, Xgboost, но в такой задаче очень маловероятно, что что-то справится лучше линейных методов
# - интересно проверить качество алгоритма на данных, где сессии выделялись не по количеству посещенных сайтов, а по времени, например, 5, 7, 10 и 15 минут. Отдельно стоит отметить данные нашего [соревнования](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) 
# - опять же, если ресурсы позволяют, можно проверить, насколько хорошо можно решить задачу для 3000 пользователей
# 
# 
# На следующей неделе мы вспомним про линейные модели, обучаемые стохастическим градиентным спуском, и порадуемся тому, насколько быстрее они работают. Также сделаем первые (или не первые) посылки в [соревновании](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) Kaggle Inclass.
