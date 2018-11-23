
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/web/677/8e1/337/6778e1337c3d4b159d7e99df94227cb2.jpg"/>
# ## Специализация "Машинное обучение и анализ данных"
# <center>Автор материала: программист-исследователь Mail.Ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ [Юрий Кашницкий](https://yorko.github.io/)

# # <center> Capstone проект №1 <br> Идентификация пользователей по посещенным веб-страницам
# <img src='http://i.istockimg.com/file_thumbview_approve/21546327/5/stock-illustration-21546327-identification-de-l-utilisateur.jpg'>
# 
# # <center>Неделя 6.  Vowpal Wabbit
# 
# На этой неделе мы познакомимся с популярной библиотекой Vowpal Wabbit и попробуем ее на данных по посещению сайтов.
# 
# **План 6 недели:**
# - Часть 1. Статья по Vowpal Wabbit
# - Часть 2. Применение Vowpal Wabbit к данным по посещению сайтов
#  - 2.1. Подготовка данных
#  - 2.2. Валидация по отложенной выборке
#  - 2.3. Валидация по тестовой выборке (Public Leaderboard)
# 
# **В этой части проекта Вам могут быть полезны видеозаписи следующих лекций курса "Обучение на размеченных данных":**
#    - [Стохатический градиентный спуск](https://www.coursera.org/learn/supervised-learning/lecture/xRY50/stokhastichieskii-ghradiientnyi-spusk)
#    - [Линейные модели. `sklearn.linear_model`. Классификация](https://www.coursera.org/learn/supervised-learning/lecture/EBg9t/linieinyie-modieli-sklearn-linear-model-klassifikatsiia)
#    
# Также будет полезна [презентация](https://github.com/esokolov/ml-course-msu/blob/master/ML15/lecture-notes/Sem08_vw.pdf) лектора специализации Евгения Соколова. И, конечно же, [документация](https://github.com/JohnLangford/vowpal_wabbit/wiki) Vowpal Wabbit.

# ### Задание
# 1. Заполните код в этой тетрадке 
# 2. Если вы проходите специализацию Яндеса и МФТИ, пошлите файл с ответами в соответствующем Programming Assignment. <br> Если вы проходите курс ODS, выберите ответы в [веб-форме](https://docs.google.com/forms/d/1wteunpEhAt_9s-WBwxYphB6XpniXsAZiFSNuFNmvOdk).

# ## Часть 1. Статья про Vowpal Wabbit
# Прочитайте [статью](https://habrahabr.ru/company/ods/blog/326418/) про Vowpal Wabbit на Хабре из серии открытого курса OpenDataScience по машинному обучению. Материал для этой статьи зародился из нашей специализации. Скачайте [тетрадку](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_russian/topic08_sgd_hashing_vowpal_wabbit/topic8_sgd_hashing_vowpal_wabbit.ipynb), прилагаемую к статье, посмотрите код, изучите его, поменяйте, только так можно разобраться с Vowpal Wabbit.

# ## Часть 2. Применение Vowpal Wabbit к данным по посещению сайтов

# ### 2.1. Подготовка данных

# **Далее посмотрим на Vowpal Wabbit в деле. Правда, в задаче нашего соревнования при бинарной классификации веб-сессий мы разницы не заметим – как по качеству, так и по скорости работы (хотя можете проверить), продемонстрируем всю резвость VW в задаче классификации на 400 классов. Исходные данные все те же самые, но выделено 400 пользователей, и решается задача их идентификации. Скачайте данные [отсюда](https://inclass.kaggle.com/c/identify-me-if-you-can4/data) – файлы `train_sessions_400users.csv` и `test_sessions_400users.csv`.**

# In[1]:


import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier


# In[2]:


# Поменяйте на свой путь к данным
PATH_TO_DATA = 'capstone_user_identification'


# **Загрузим обучающую и тестовую выборки. Можете заметить, что тестовые сессии здесь по времени четко отделены от сессий в обучающей выборке. **

# In[3]:


train_df_400 = pd.read_csv(os.path.join(PATH_TO_DATA,'train_sessions_400users.csv'), 
                           index_col='session_id')


# In[4]:


test_df_400 = pd.read_csv(os.path.join(PATH_TO_DATA,'test_sessions_400users.csv'), 
                           index_col='session_id')


# In[5]:


train_df_400.head()


# **Видим, что в обучающей выборке 182793 сессий, в тестовой – 46473, и сессии действительно принадлежат 400 различным пользователям.**

# In[6]:


train_df_400.shape, test_df_400.shape, train_df_400['user_id'].nunique()


# **Vowpal Wabbit любит, чтоб метки классов были распределены от 1 до K, где K – число классов в задаче классификации (в нашем случае – 400). Поэтому придется применить `LabelEncoder`, да еще и +1 потом добавить (`LabelEncoder` переводит метки в диапозон от 0 до K-1). Потом надо будет применить обратное преобразование.**

# In[ ]:


y = ''' ВАШ КОД ЗДЕСЬ '''
class_encoder = ''' ВАШ КОД ЗДЕСЬ '''
y_for_vw = ''' ВАШ КОД ЗДЕСЬ '''


# **Далее будем сравнивать VW с SGDClassifier и с логистической регрессией. Всем моделям этим нужна предобработка входных данных. Подготовьте для sklearn-моделей разреженные матрицы, как мы это делали в 5 части:**
# - объедините обучающиую и тестовую выборки
# - выберите только сайты (признаки от 'site1' до 'site10')
# - замените пропуски на нули (сайты у нас нумеровались с 0)
# - переведите в разреженный формат `csr_matrix`
# - разбейте обратно на обучающую и тестовую части

# In[ ]:


sites = ['site' + str(i) for i in range(1, 11)]


# In[ ]:


''' ВАШ КОД ЗДЕСЬ '''
X_train_sparse = ''' ВАШ КОД ЗДЕСЬ '''
X_test_sparse = ''' ВАШ КОД ЗДЕСЬ '''
y = ''' ВАШ КОД ЗДЕСЬ '''


# ### 2.2. Валидация по отложенной выборке

# **Выделим обучающую (70%) и отложенную (30%) части исходной обучающей выборки. Данные не перемешиваем, учитываем, что сессии отсортированы по времени.**

# In[ ]:


train_share = int(.7 * train_df_400.shape[0])
train_df_part = train_df_400[sites].iloc[:train_share, :]
valid_df = train_df_400[sites].iloc[train_share:, :]
X_train_part_sparse = X_train_sparse[:train_share, :]
X_valid_sparse = X_train_sparse[train_share:, :]


# In[ ]:


y_train_part = y[:train_share]
y_valid = y[train_share:]
y_train_part_for_vw = y_for_vw[:train_share]
y_valid_for_vw = y_for_vw[train_share:]


# **Реализуйте функцию, `arrays_to_vw`, переводящую обучающую выборку в формат Vowpal Wabbit.**
# 
# Вход:
#  - X – матрица `NumPy` (обучающая выборка)
#  - y (необяз.) - вектор ответов (`NumPy`). Необязателен, поскольку тестовую матрицу будем обрабатывать этой же функцией
#  - train – флаг, True в случае обучающей выборки, False – в случае тестовой выборки
#  - out_file – путь к файлу .vw, в который будет произведена запись
#  
# Детали:
# - надо пройтись по всем строкам матрицы `X` и записать через пробел все значения, предварительно добавив вперед нужную метку класса из вектора `y` и знак-разделитель `|`
# - в тестовой выборке на месте меток целевого класса можно писать произвольные, допустим, 1

# In[ ]:


def arrays_to_vw(X, y=None, train=True, out_file='tmp.vw'):
    ''' ВАШ КОД ЗДЕСЬ '''
    pass


# **Примените написанную функцию к части обучащей выборки `(train_df_part, y_train_part_for_vw)`, к отложенной выборке `(valid_df, y_valid_for_vw)`, ко всей обучающей выборке и ко всей тестовой выборке. Обратите внимание, что на вход наш метод принимает именно матрицы и вектора `NumPy`.**

# In[ ]:


get_ipython().run_cell_magic('time', '', "# будет 4 вызова\narrays_to_vw ''' ВАШ КОД ЗДЕСЬ '''")


# **Результат должен получиться таким.**

# In[7]:


get_ipython().system('head -3 $PATH_TO_DATA/train_part.vw')


# In[8]:


get_ipython().system('head -3  $PATH_TO_DATA/valid.vw')


# In[9]:


get_ipython().system('head -3 $PATH_TO_DATA/test.vw')


# **Обучите модель Vowpal Wabbitна выборке `train_part.vw`. Укажите, что решается задача классификации с 400 классами (`--oaa`), сделайте 3 прохода по выборке (`--passes`). Задайте некоторый кэш-файл (`--cache_file`, можно просто указать флаг `-c`), так VW будет быстрее делать все следующие после первого проходы по выборке (прошлый кэш-файл удаляется с помощью аргумента `-k`). Также укажите значение параметра `b`=26. Это число бит, используемых для хэширования, в данном случае нужно больше, чем 18 по умолчанию. Наконец, укажите `random_seed`=17. Остальные параметры пока не меняйте, далее уже в свободном режиме соревнования можете попробовать другие функции потерь.**

# In[ ]:


train_part_vw = os.path.join(PATH_TO_DATA, 'train_part.vw')
valid_vw = os.path.join(PATH_TO_DATA, 'valid.vw')
train_vw = os.path.join(PATH_TO_DATA, 'train.vw')
test_vw = os.path.join(PATH_TO_DATA, 'test.vw')
model = os.path.join(PATH_TO_DATA, 'vw_model.vw')
pred = os.path.join(PATH_TO_DATA, 'vw_pred.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "!vw ''' ВАШ КОД ЗДЕСЬ '''")


# **Запишите прогнозы на выборке *valid.vw* в *vw_valid_pred.csv*.**

# In[ ]:


get_ipython().run_cell_magic('time', '', "!vw ''' ВАШ КОД ЗДЕСЬ '''")


# **Считайте прогнозы *kaggle_data/vw_valid_pred.csv*  из файла и посмотрите на долю правильных ответов на отложенной части.**

# In[ ]:


''' ВАШ КОД ЗДЕСЬ '''


# **Теперь обучите `SGDClassifier` (3 прохода по выборке, логистическая функция потерь) и `LogisticRegression` на 70% разреженной обучающей выборки – `(X_train_part_sparse, y_train_part)`, сделайте прогноз для отложенной выборки `(X_valid_sparse, y_valid)` и посчитайте доли верных ответов. Логистическая регрессия будет обучаться не быстро (у меня – 4 минуты) – это нормально. Укажите везде `random_state`=17, `n_jobs`=-1. Для `SGDClassifier` также укажите `max_iter=3`.**

# In[ ]:


logit = ''' ВАШ КОД ЗДЕСЬ '''
sgd_logit = ''' ВАШ КОД ЗДЕСЬ '''


# In[ ]:


get_ipython().run_cell_magic('time', '', "logit.fit ''' ВАШ КОД ЗДЕСЬ '''")


# In[ ]:


get_ipython().run_cell_magic('time', '', "sgd_logit.fit ''' ВАШ КОД ЗДЕСЬ '''")


# **<font color='red'>Вопрос 1. </font> Посчитайте долю правильных ответов на отложенной выборке для Vowpal Wabbit, округлите до 3 знаков после запятой.**
# 
# **<font color='red'>Вопрос 2. </font> Посчитайте долю правильных ответов на отложенной выборке для SGD, округлите до 3 знаков после запятой.**
# 
# **<font color='red'>Вопрос 3. </font> Посчитайте долю правильных ответов на отложенной выборке для логистической регрессии, округлите до 3 знаков после запятой.**

# In[ ]:


vw_valid_acc = ''' ВАШ КОД ЗДЕСЬ '''
sgd_valid_acc = ''' ВАШ КОД ЗДЕСЬ '''
logit_valid_acc = ''' ВАШ КОД ЗДЕСЬ '''


# In[ ]:


def write_answer_to_file(answer, file_address):
    with open(file_address, 'w') as out_f:
        out_f.write(str(answer))


# In[ ]:


write_answer_to_file(round(vw_valid_acc, 3), 'answer6_1.txt')
write_answer_to_file(round(sgd_valid_acc, 3), 'answer6_2.txt')
write_answer_to_file(round(logit_valid_acc, 3), 'answer6_3.txt')


# ### 2.3. Валидация по тестовой выборке (Public Leaderboard)

# **Обучите модель VW с теми же параметрами на всей обучающей выборке – *train.vw*.**

# In[ ]:


get_ipython().run_cell_magic('time', '', "!vw ''' ВАШ КОД ЗДЕСЬ '''")


# **Сделайте прогноз для тестовой выборки.**

# In[ ]:


get_ipython().run_cell_magic('time', '', "!vw ''' ВАШ КОД ЗДЕСЬ '''")


# **Запишите прогноз в файл, примените обратное преобразование меток (был LabelEncoder и потом +1 в меткам) и отправьте решение на Kaggle.**

# In[ ]:


def write_to_submission_file(predicted_labels, out_file,
                             target='user_id', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[ ]:


vw_pred = ''' ВАШ КОД ЗДЕСЬ '''


# In[ ]:


write_to_submission_file(vw_pred, os.path.join(PATH_TO_DATA, 'vw_400_users.csv'))


# **Сделайте то же самое для SGD и логистической регрессии. Тут уже ждать обучение логистической регрессии совсем скучно (заново запускать тетрадку вам не захочется), но давайте дождемся.**

# In[ ]:


''' ВАШ КОД ЗДЕСЬ '''


# In[ ]:


write_to_submission_file(sgd_logit_test_pred, 
                         os.path.join(PATH_TO_DATA, 'logit_400_users.csv'))
write_to_submission_file(sgd_logit_test_pred, 
                         os.path.join(PATH_TO_DATA, 'sgd_400_users.csv'))


# Посмотрим на доли правильных ответов на публичной части (public leaderboard) тестовой выборки [этого](https://inclass.kaggle.com/c/identify-me-if-you-can4) соревнования.
# 
# **<font color='red'>Вопрос 4. </font> Какова доля правильных ответов на публичной части тестовой выборки (public leaderboard)  для Vowpal Wabbit?**
# 
# **<font color='red'>Вопрос 5. </font> Какова доля правильных ответов на публичной части тестовой выборки (public leaderboard)  для SGD?**
# 
# **<font color='red'>Вопрос 6. </font> Какова доля правильных ответов на публичной части тестовой выборки (public leaderboard)  для логистической регрессии?**
# 

# In[ ]:


vw_lb_score, sgd_lb_score, logit_lb_score = ''' ВАШ КОД ЗДЕСЬ '''

write_answer_to_file(round(vw_lb_score, 3), 'answer6_4.txt')
write_answer_to_file(round(sgd_lb_score, 3), 'answer6_5.txt')
write_answer_to_file(round(logit_lb_score, 3), 'answer6_6.txt')


# **В заключение по заданию:**
# - Про соотношение качества классификации и скорости обучения VW, SGD и logit выводы предлагается сделать самостоятельно
# - Пожалуй, задача классификации на 400 классов (идентификация 400 пользователей) решается недостаточно хорошо при честном отделении по времени тестовой выборки от обучающей. Далее мы будем соревноваться в идентификации одного пользователя (Элис) – [вот](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) соревнование, в котором предлагается поучаствовать. Не перепутайте! 
# 
# **Удачи!**
