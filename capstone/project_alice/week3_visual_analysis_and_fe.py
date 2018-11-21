
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/web/677/8e1/337/6778e1337c3d4b159d7e99df94227cb2.jpg"/>
# ## Специализация "Машинное обучение и анализ данных"
# <center>Автор материала: программист-исследователь Mail.Ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ [Юрий Кашницкий](https://yorko.github.io/)

# # <center>  Capstone проект №1 <br> Идентификация пользователей по посещенным веб-страницам
# <img src='http://i.istockimg.com/file_thumbview_approve/21546327/5/stock-illustration-21546327-identification-de-l-utilisateur.jpg'>
# 
# # <center>Неделя 3. Визуальный анализ данных и построение признаков
# 
# На 3 неделе мы займемся визуальным анализом данных и построением признаков. Сначала мы вместе построим и проанализируем несколько признаков, потом Вы сможете сами придумать и описать различные признаки. 
# 
# **План 3 недели:**
#  - Часть 1. Построение признаков
#  - Часть 2. Визуальный анализ данных
#  - Часть 3. Дальнейшее построение признаков
#  - Часть 4. Проверка построенных признаков
# 
# **В этой части проекта Вам могут быть полезны видеозаписи следующих лекций курса "Поиск структуры в данных":**
#    - [Задача визуализации](https://www.coursera.org/learn/unsupervised-learning/lecture/hlvlT/zadacha-vizualizatsii)
#    - [Визуализация данных в sklearn](https://www.coursera.org/learn/unsupervised-learning/lecture/ityMo/vizualizatsiia-dannykh-v-sklearn)
#    
# **Также в задании будет использоваться библиотека Seaborn (ее можно дополнительно установить командой *pip install seaborn*), будет полезно обращаться к документациям [Matplotlib](http://matplotlib.org/users/) и [Seaborn](http://seaborn.pydata.org/), а также к примерам визуализации, описанным на StackOverflow.**
# 
# 

# ### Задание
# 1. Заполните код в этой тетрадке 
# 2. Если вы проходите специализацию Яндеса и МФТИ, пошлите тетрадку в соответствующем Peer Review. <br> Если вы проходите курс ODS, выберите ответы в [веб-форме](https://docs.google.com/forms/d/1EbjK7-hF-Gepi6RH-K5I2XeiYGRoY0LNDx03QmLu9Xo). 

# ## Часть 1. Построение признаков

# In[1]:


from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import re
import os
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
pd.set_option('display.max.columns', 25)
from scipy.sparse import csr_matrix
import pickle
#pip install seaborn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['patch.force_edgecolor'] = True


# In[2]:


# Поменяйте на свой путь к данным
PATH_TO_DATA = 'data'


# **Создайте на основе функций *prepare_train_set* и *prepare_sparse_train_set_window*  новую – *prepare_train_set_with_fe*, (от "feature engineering"), создайте следующие признаки:**
# - `session_timespan` – продолжительность сессии (разница между максимальным и минимальным временем посещения сайтов в сессии, в секундах)
# - `#unique_sites` – число уникальных сайтов в сессии 
# - `start_hour` – час начала сессии (то есть час в записи минимального timestamp среди десяти)
# - `day_of_week` – день недели (то есть день недели в записи минимального timestamp среди десяти)
# 
# Функция должна возвращать новый DataFrame (как возвращала функция *prepare_train_set*), только признаков должно быть на 4 больше. Порядок, в котором добавляются признаки: *site1*, ... *site10*, *session_timespan*, *#unique_sites*, *start_hour*, *day_of_week* и *user_id* (это видно и чуть ниже по тому, как функция вызывается).

# In[3]:


def get_sites_dict(path_to_csv_files):
    """Returns sites visit frequency dictionary
    
    Form of the dictonary is ``{'sitename': [site_id, frequency]}``.
    """
    
    arr = []
    '''Pull all visited sites from .csv logs to array'''
    for file in glob(path_to_csv_files + '/*'):
        with open(file) as f:
            lines = f.read().split('\n')
            for line in lines[1:-1]:  # Exclude header
                arr.append(line.split(',')[1])
                
    '''Make counted sorted dict with sites and corresponding # of visits (freqency)'''
    sites_freq_list = sorted(Counter(arr).items(), key=lambda x: x[1], reverse=True)
    '''Make sites dict in form of {'site': [id, frequency]}'''
    sites_dict = dict((s, [i, freq]) for i, (s, freq) in enumerate(sites_freq_list, 1))
    '''Make id to site converter'''
    id_to_site_dict = dict((val[0], key) for key, val in sites_dict.items())
    id_to_site_dict[0] = 'no_site'

    return sites_dict, id_to_site_dict


# In[4]:


def to_csr(X):
    session_length = X.shape[1]
    data = [1] * X.ravel().shape[0]
    indices = X.ravel()
    indptr = range(0, X.ravel().shape[0] + session_length, session_length)
    return csr_matrix((data, indices, indptr))[:, 1:]


# In[5]:


def prepare_train_set_with_fe(path_to_csv_files, session_length=10,
                              window_size=10, return_table=False):
    """Func for partition users logs to desireable num of sessions
        and prepare training dataset with sessions of particular users.
    """
    
    df = []
    
    sites_dict, _ = get_sites_dict(path_to_csv_files)
    
    for file in tqdm(glob(path_to_csv_files + '/*'), desc='Preparing training set...'):
            
        temp_df = pd.read_csv(file, parse_dates=['timestamp'], engine='c')
        temp_df['site_id'] = temp_df.site.apply(lambda x: sites_dict[x][0])
        
        '''Feature generating with sliding window'''
        windptr = range(0, temp_df.shape[0], window_size)
        sessions = np.array([])  # Sites in a session row
        timedeltas = []  # Time of the session
        uniques = []  # Number of unique sites in the session
        start_hours = []  # Session start hour
        days_of_week = []  # Session start weekday
        for ptr in windptr:
            sess = temp_df.site_id.values[ptr:ptr+session_length]
            if len(sess) < session_length:  # All incomplete session windows are being completed by zeros
                sess = np.r_[sess, [0] * (session_length - len(sess))]
            
            sessions = np.hstack((sessions, sess))
            
            uniques.append(len(set(sess)))
            
            timespan = temp_df.timestamp.values[ptr:ptr+session_length]
            timedeltas.append((timespan[-1] - timespan[0]) / np.timedelta64(1, 's'))
            
            start_time = timespan[0].astype('datetime64[h]').astype(object)
            start_hours.append(start_time.hour)
            
            days_of_week.append(start_time.weekday())
            
        '''Preparing the columns'''
        sessions = sessions.reshape(-1, session_length)  # Partition user data to sessions
        timedeltas = np.asarray(timedeltas, dtype=np.int32).reshape(sessions.shape[0], 1)
        uniques = np.asarray(uniques, dtype=np.int8).reshape(sessions.shape[0], 1)
        start_hours = np.asarray(start_hours, dtype=np.int8).reshape(sessions.shape[0], 1)
        days_of_week = np.asarray(days_of_week, dtype=np.int8).reshape(sessions.shape[0], 1)
        
        '''Build a full dataset consisting of user id's and sessions'''
        feature_names = ['site'+ str(x+1) for x in range(session_length)] +                         ['session_timespan', '#unique_sites', 'start_hour', 
                         'day_of_week']
        temp_df = pd.DataFrame(np.hstack((sessions, timedeltas, uniques, start_hours, days_of_week)),
                               columns=feature_names, dtype=np.int32)
        
        '''Find user's id [target feature] in the file name'''
        user_id = re.findall(r'\d+', file)[-1]
        temp_df['user_id'] = [int(user_id)] * temp_df.shape[0]
        df.append(temp_df)
        
    df = pd.concat(df, ignore_index=True)
        
    if return_table:
        return df
    else:
        X_sites, X_add, y = df.iloc[:, :-5].values, df.iloc[:, -5:-1].values, df.iloc[:, -1].values
        return to_csr(X_sites), X_add, y


# **Проверим функцию на игрушечном примере.**
X_sites, X_add, y = prepare_train_set_with_fe(
    os.path.join(PATH_TO_DATA, '150users'), session_length=10, window_size=5)
# In[6]:


train_data_toy  = prepare_train_set_with_fe(
    os.path.join(PATH_TO_DATA, '3users'), 
    session_length=10, 
    window_size=5, 
    return_table=True
)

train_data_toy


# **Примените функцию *prepare_train_set_with_fe* к данным по 10 пользователям, укажите *session_length*=10.**

# In[7]:


train_data_10users = prepare_train_set_with_fe(os.path.join(PATH_TO_DATA, '10users'),
                                               return_table=True)


# In[8]:


train_data_10users.head()


# **Примените функцию *prepare_train_set_with_fe* к данным по 150 пользователям, укажите *session_length*=10.**

# In[9]:


train_data_150users = prepare_train_set_with_fe(os.path.join(PATH_TO_DATA, '150users'),
                                                return_table=True)


# **Сохраните в pickle-файлы признаки *session_timespan*, *#unique_sites*, *start_hour* и *day_of_week* для 10 и 150 пользователей.**

# In[10]:


new_features_10users = train_data_10users.iloc[:, -5:-1]
new_features_150users = train_data_150users.iloc[:, -5:-1]


# In[11]:


with open(os.path.join(PATH_TO_DATA, 
                       'new_features_10users.pkl'), 'wb') as new_features_10users_pkl:
    pickle.dump(new_features_10users, new_features_10users_pkl)
with open(os.path.join(PATH_TO_DATA, 
                       'new_features_150users.pkl'), 'wb') as new_features_150users_pkl:
    pickle.dump(new_features_150users, new_features_150users_pkl)


# **<font color='red'>Вопрос 1. </font> Выведите медианную продолжительность сессии (*session_timespan*) для сессий 10 пользователей.**

# In[12]:


new_features_10users.describe().astype(int).loc['50%']['session_timespan']


# **<font color='red'>Вопрос 2. </font> Выведите медианный день недели, в который началась сессия, для сессий 10 пользователей.**

# In[13]:


print('2 is Wednesday')
new_features_10users.describe().astype(int).loc['50%']['day_of_week']


# **<font color='red'>Вопрос 3. </font>Выведите медианный час начала сессии для сессий 150 пользователей.**

# In[14]:


new_features_150users.describe().astype(int).loc['50%']['start_hour']


# **<font color='red'>Вопрос 4. </font>Выведите медианное значение числа уникальных сайтов в сессиях 150 пользователей.**

# In[15]:


new_features_150users.describe().astype(int).loc['50%']['#unique_sites']


# ## Часть 2. Визуальный анализ данных

# **Забавы ради, потехи для дадим пользователям имена и ассоциируем с ними цвета.**

# In[16]:


id_name_dict = {128: 'Mary-Kate', 39: 'Ashley', 207: 'Lindsey', 127: 'Naomi', 237: 'Avril',
               33: 'Bob', 50: 'Bill', 31: 'John', 100: 'Dick', 241: 'Ed'}
train_data_10users['user_id'] = train_data_10users['user_id'].map(id_name_dict)


# In[17]:


import matplotlib.cm as cm
user_list = train_data_10users.user_id.unique()
color_dict = dict(
    (user, color) for user, color in zip(user_list, cm.rainbow_r(np.linspace(0, 1, len(user_list))))
)


# In[18]:


train_data_10users.head()


# **1. Постройте гистограмму распределения длины сессии в секундах (*session_timespan*). Ограничьте по *x* значением 200 (иначе слишком тяжелый хвост). Сделайте гистограмму цвета *darkviolet*, подпишите оси по-русски.**

# In[19]:


hist_data = train_data_10users['session_timespan']
plt.hist(hist_data, bins=50, range=(0,200), color='darkviolet', zorder=2)
plt.grid(True, alpha=.2, zorder=0)
plt.xlabel('Длительность сессии, с')
plt.ylabel('Количество сессий')
plt.title('Распределение длительности сессий');
plt.savefig('figs/sessions_overall.pdf')


# **2. Постройте гистограмму распределения числа уникальных сайтов в сессии (*#unique_sites*). Сделайте гистограмму цвета *aqua*, подпишите оси по-русски.**

# В случаях, когда уникальные значения дискретны и их количество невелико, для подсчета количества сессий по каждому значению удобнее использовать `seaborn.countplot()`

# In[20]:


hist_data = train_data_10users['#unique_sites']
s = sns.countplot(hist_data, color='aqua', zorder=2)
s.set_xlabel('Число уникальных сайтов в сессии')
s.set_ylabel('Количество сессий')
plt.grid(True, alpha=.2, zorder=0, axis='y')
plt.title('Распределение числа\nуникальных сайтов в сессиях');
plt.savefig('figs/#unique_sites_overall.pdf')


# **3. Постройте гистограммы распределения числа уникальных сайтов в сессии (*#unique_sites*) для каждого из 10 пользователей по отдельности. Используйте *subplots*, чтоб разместить все 10 картинок на одной большой. Пометьте легендой каждую картинку, на легенде должно быть написано имя пользователя. Для каждого пользователя раскрасьте гистограмму его/ее цветом (*color_dic*). Подпишите оси по-русски в каждой из 10 гистограмм.**

# In[21]:


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))
colors = cm.rainbow_r(np.linspace(0, 1, 10))
for idx, (user, sub_df) in enumerate(pd.groupby(train_data_10users, 'user_id')):
    s = sns.countplot(sub_df['#unique_sites'], color=color_dict[user], saturation=.4,
                      ax=ax[idx//4, idx%4], label=user, zorder=2)
    s.set_xlabel('Число уникальных сайтов в сессии')
    s.set_ylabel('Количество сессий')
    ax[idx//4, idx%4].grid(True, alpha=.2, zorder=0, axis='y')
    s.legend()
    
    
suptitle = fig.suptitle('Распределение количества уникальных сайтов в сессии для всех пользователей', y=1.03,
             fontsize=16)
fig.tight_layout();
fig.savefig('figs/#unique_sites_per_user.pdf', bbox_inches='tight', bbox_extra_artists=[suptitle])


# **4. Постройте гистограмму распределения часа начала сессии (*start_hour*). Сделайте гистограмму цвета *darkgreen*, подпишите оси по-русски.**

# In[22]:


hist_data = train_data_10users['start_hour']
s = sns.countplot(hist_data, color='darkgreen', saturation=1, zorder=2)
s.set_xlabel('Час начала сессии')
s.set_ylabel('Количество сессий')
plt.grid(True, alpha=.2, zorder=0, axis='y')
plt.title('Распределение времени начала сессий');
plt.savefig('figs/start_hour_overall.pdf')


# **5. Постройте гистограммы распределения часа начала сессии (*start_hour*) для каждого из 10 пользователей по отдельности. Используйте *subplots*, чтоб разместить все 10 картинок на одной большой. Пометьте легендой каждую картинку, на легенде должно быть написано имя пользователя. Для каждого пользователя раскрасьте гистограмму его/ее цветом (*color_dic*). Подпишите оси по-русски в каждой из 10 гистограмм.**

# In[23]:


import matplotlib.cm as cm
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))
colors = cm.Greens(np.linspace(0, 1, 12))
for idx, (user, sub_df) in enumerate(pd.groupby(train_data_10users, 'user_id')):
    s = sns.countplot(sub_df.start_hour, color=color_dict[user], saturation=.4, ax=ax[idx//4, idx%4], label=user,
                      zorder=2)
    ax[idx//4, idx%4].grid(True, alpha=.2, zorder=0, axis='y')
    s.set_xlabel('Час начала сессии')
    s.set_ylabel('Количество сессий')
    s.legend()
    
suptitle = fig.suptitle('Распределение времени старта сессии для всех пользователей', y=1.03,
             fontsize=16)
fig.tight_layout();
fig.savefig('figs/start_hour_per_user.pdf', bbox_inches='tight', bbox_extra_artists=[suptitle])


# **6. Постройте гистограмму распределения дня недели, в который началась сессия (*day_of_week*). Сделайте гистограмму цвета *sienna*, подпишите оси по-русски.**

# In[24]:


hist_data = train_data_10users['day_of_week']
s = sns.countplot(hist_data, color='sienna', zorder=2)
plt.grid(True, alpha=.2, zorder=0, axis='y')
s.set_xlabel('День недели')
s.set_ylabel('Количество сессий')
s.set_xticklabels(['Пн','Вт','Ср','Чт','Пт','Сб','Вс'])
s.set_title('Распределение сессий по дням недели');
plt.savefig('figs/day_of_week_overall.pdf')


# **7. Постройте гистограммы распределения дня недели, в который началась сессия (*day_of_week*) для каждого из 10 пользователей по отдельности. Используйте *subplots*, чтоб разместить все 10 картинок на одной большой. Измените метки по оси *X* на ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'] – метод *set_xticklabels*. Пометьте легендой каждую картинку, на легенде должно быть написано имя пользователя. Для каждого пользователя раскрасьте гистограмму его/ее цветом (*color_dic*). Подпишите по-русски название каждой из 10 гистограмм.**

# In[25]:


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

# как вариант, можно и по-другому
for idx, (user, sub_df) in  enumerate(pd.groupby(train_data_10users, 'user_id')):
    s = sns.countplot(sub_df.day_of_week, color=color_dict[user], saturation=.4, ax=ax[idx//4, idx%4], label=user,
                      zorder=2)
    ax[idx//4, idx%4].grid(True, alpha=.2, zorder=0, axis='y')
    s.set_xlabel('День недели')
    s.set_ylabel('Количество сессий')
    s.legend()
    s.set_xticklabels(['Пн','Вт','Ср','Чт','Пт','Сб','Вс'])
    s.set_title('Распределение сессий по дням недели', fontsize=10)

suptitle = fig.suptitle('Распределение сессий по дням недели для каждого пользователя', y=1.03, fontsize=16)
fig.tight_layout();
fig.savefig('figs/day_of_week_per_user.pdf', bbox_inches='tight', bbox_extra_artist=[suptitle])


# **8. Сделайте выводы про каждого пользователя по построенным графикам.**

# Мы оценим характерные особенности каждого пользователя.
# #### Для пользователя характерно следующее:
#  - **Ashley**:
#    - Утренние и дневные сессии. Почти полное отсутствие вечерних. 
#    - Большая часть сессий в среду, остутствие сессий в выходные дни. (Рабочий компьютер?)
#    - Большое количество сессий с разными сайтами (возможно большой стек привычных сайтов). Много сессий с одним сайтом (гугл?)
#  - **Avril**
#    - Дневные и вечерние сессии. Мало вечерних сессий.
#    - Мало сессий в Пт и Сб.
#    - 
#    
#  - **Bill**
#    - Утренние и дневные сессии. Почти полное отсутствие обеденных и вечерних сессий. 
#    - Меньше сессий во второй половине недели.
#    - Среднее количество уникальных сайтов в сессии  (7 – как у болшинства в статистике по всем пользователям). Много сессий с одним сайтом (гугл?)
#  
#  - **Bob**
#    - Утренние и дневные сессии. 
#    - Отсутстиве сессий на выходных (Рабочий компьютер?). Мало сессий в Пн и Ср.
#    - Среднее количество уникальных сайтов в сессии.
#  
#  - **Dick**
#    - Стабильные сессии в течение всего дня за исключением времени обеда и ужина.
#    - Почти полное отсутствие сессий в Пн и Ср. Много в Пт. Нет сессий в Вс.
#    - Среднее количество уникальных сайтов в сессии. Много сессий с двумя уникальными сайтами (возможно, youtube, для него в словаре сайтов несколько ссылок)
#  
#  - **Ed**
#    - В основном дневные сессии. 
#    - Мало сессий в Пн, Вт и Чт. Много в Ср.
#    - Среднее количество уникальных сайтов в сессии.
#  
#  - **John**
#    - Стабильные сессии в течение дня и почти полное отстутствие вечерних сессий.
#    - Нет сессий в Вс. Исчезающе мало в Сб.
#    - Среднее количество уникальных сайтов в сессии.
#  
#  - **Lindsey**
#    - Дневные сессии и почти полное отсутствие вечерних.
#    - Основное количество сессий во Вт и Ср. Мало – в Сб, Вс.
#    - Среднее количество уникальных сайтов в сессии.
#  
#  - **Mary-Kate**
#    - Стабильные сессии в течение всего дня за исключением раннего утра и позднего вечера.
#    - Сессий на выходных в среднем сильно больше чем в течение недели.
#    - Много сессий с двумя уникальными сайтами (возможно, youtube, для него в словаре сайтов несколько ссылок).
#  
#  - **Naomi**
#    - Дневные сессии и почти полное отсутствие сессий во время ужина.
#    - Больше сессий в четверг.
#    - Среднее количество уникальных сайтов в сессии.

# **Загрузите сохраненный ранее в pickle-файл частотный словарь сайтов для 10 пользователей. **

# In[26]:


with open('data/site_freq_150users.pkl', 'rb') as f:
    site_freq_10users = pickle.load(f)
site_freq_10users = pd.DataFrame.from_dict(site_freq_10users, orient='index')
site_freq_10users.columns = ['site_id', 'frequency']
site_freq_10users['site'] = site_freq_10users.index
site_freq_10users = site_freq_10users.set_index('site_id')
site_freq_10users_top10 = site_freq_10users.sort_values(by='frequency', ascending=False).head(10)
site_freq_10users_top10


# **Определите топ-10 самых посещаемых сайтов (*top10_sites*) и соответствующие кол-ва посещений (*top10_freqs*).**

# In[28]:


top10_freqs = site_freq_10users_top10.frequency.values
top10_sites = site_freq_10users_top10.site.values


# **9. Нарисуйте *seaborn barplot*, показывающий частоты посещений топ-10 сайтов. Сделайте подписи сайтов вертикальными, иначе они сливаются (*xticks*).**

# In[29]:


plt.figure(figsize=(16,4))
s = sns.barplot(data=site_freq_10users_top10, x='site', y='frequency', color='blue', saturation=.2)
s.set_xticklabels(s.get_xticklabels(), rotation=10);


# ## Часть 3. Дальнейшее построение признаков

# Это задание творческое, тут надо придумать, как еще учесть время посещения веб-страниц и прочие признаки. 
# 
# На следующей неделе мы будем использовать "мешок" сайтов  для классификации сессий по принадлежности разным пользователям, а эти новые признаки, которые Вы сейчас создадите, потом добавим и посмотрим, улучшается ли модель. Поэтому можно их создать в виде отдельных матриц и сохранить их также отдельно.  
# 
# В этой части задания Вы можете построить и визуально исследовать самые разные признаки (ничто фантазию не ограничивает):
# - год, месяц и день начала сессии
# - час начала сессии (с учетом года, месяца и дня)
# - время суток
# - среднее время пребывания на сайте, посчитать можно, скажем, для топ-30 популярных сайтов
# - индикаторы посещения популярных сайтов (скажем, тоже для топ-30 популярных сайтов)
# - частота посещения Facebook
# - ...

# _**Напишите функцию для создания новых признаков и примените ее к исходным данным – каталогам с 10 и 150 файлами. Сделайте это только для набора данных, полученного с параметрами *session_length=10* и *window_size=10*. Сериализуйте полученные матрицы с помощью pickle. Функция может возвращать как только новые признаки, так и старые с новыми. При этом сигнатура функции может быть другой – тут уже свобода выбора.**_

# ### Конструирование новых признаков

# На мой взгляд признаки связанные с посещаемостью сайтов уже заложены в sparse матрицах. Самые популярные сайты, наиболее характерные сайты можно будет выделить с помощью конструирования признаков с использованием, например `PCA` или матричного разложения. Поэтому на данном этапе будем конструировать признаки на основе метое времени посещения сайта. 
# 
# **Вышеприведенные гистограммы не дают нам полного представления о паттерне поведения пользователя** в течение недели. При этом, из гистограмм мы видим, что **пользователю может быть характерно выходить в сеть утром или во второй половине дня, реже выходить в сеть в обеденное время, а также выходить меньше или больше в выходные дни.** Поэтому мы составим признаки, которые более точно отражают паттерн поведения пользователя в течение недели и впоследствии попробуем применить его для обучения модели.

# In[31]:


#_the_number_of_symbols_to_limit_the_line_length_according_to_PEP8_is_______79
def feature_engineering(path_to_csv_files,  session_length=10):
    """Evaluates several new features for given dataset
        
    Added daytime features, and feature which
    represents possible users's schedule: 
    Weekday + hour (of session start).
    """
    
    df = []
    sites_dict, ids_dict = get_sites_dict(path_to_csv_files)
    
    for file in tqdm(glob(path_to_csv_files + '/*'), 
                     desc='Preparing training set...'):
        user_id = int(re.findall(r'\d+', file)[-1])
        
        '''Prepare features'''
        temp_df = pd.read_csv(file, parse_dates=['timestamp'], engine='c')
        hour = temp_df['timestamp'].apply(lambda ts: ts.hour)
        temp_df['dayofweek_hour'] = temp_df.timestamp.apply(lambda x: 100*x.dayofweek) + hour
        # Daytime features
        temp_df['breakfast'] = ((hour >= 7) & (hour <= 9)).astype('int')
        temp_df['elevenses'] = ((hour > 9) & (hour <= 12)).astype('int')
        temp_df['lunch'] = ((hour > 12) & (hour <= 14)).astype('int')
        temp_df['teatime'] = ((hour > 14) & (hour <= 17)).astype('int')
        temp_df['supper'] = ((hour > 17) & (hour <= 19)).astype('int')
        temp_df['dinner'] = ((hour > 19) & (hour <= 23)).astype('int')
        
        temp_df['user_id'] = [int(user_id)] * temp_df.shape[0]
        df.append(temp_df)
        
    df = pd.concat(df, ignore_index=True)
    return df


# In[32]:


new_features_10users = feature_engineering(os.path.join(PATH_TO_DATA, '10users'))
new_features_150users = feature_engineering(os.path.join(PATH_TO_DATA, '150users'))


# **10. Постройте картинки для новых признаков, поисследуйте их, прокомментируйте результаты.**

# #### Время старта сессии с учетом дня недели

# In[33]:


new_features_10users['user_id'] = new_features_10users['user_id'].map(id_name_dict)


# In[34]:


fig, ax = plt.subplots(10, 1, figsize=(18, 32))

new_index = []
for x in range(7,707, 100):
    new_index.extend([x+y for y in range(0,17)])
    
for idx, (user, sub_df) in enumerate(new_features_10users.groupby('user_id')):
    y = sub_df.groupby('dayofweek_hour')['user_id'].agg('count').reindex(new_index).fillna(0)
    s = sns.barplot(new_index, y, color=color_dict[user], ax=ax[idx], label=user, zorder=2)
    x_labels = s.get_xticklabels()
    x_labels[0:-1:17] = ['Пн 7','Вт 7','Ср 7','Чт 7 ','Пт 7 ','Сб 7','Вс 7']
    s.set_xticklabels(x_labels, rotation=90)
    s.set_xlabel('День недели и время')
    s.set_ylabel('Количество сессий')
    s.legend()
    s.grid(axis='y', alpha=.2, zorder=0)
    
suptitle = fig.suptitle('Распределение сессий по дням и времени старта сессии', y=1.03, fontsize=16)
fig.tight_layout()
fig.savefig('figs/dayofweek_hour.pdf', bbox_inches='tight', bbox_extra_artist=[suptitle])


# #### Признаки учета утренней сессии, дневной, вечерней и тд.

# In[35]:


new_features_10users.head()


# In[36]:


daytime_cols = ['breakfast', 'elevenses', 'lunch', 'teatime', 'supper', 'dinner']
agg_df = new_features_10users.groupby('user_id')[daytime_cols].agg(np.sum)

fig, ax = plt.subplots(3, 4, figsize=(12,8))
for idx, user in enumerate(agg_df.index):
    s = sns.barplot(agg_df.loc[user].index, agg_df.loc[user].values,label=user,
                    color=color_dict[user], saturation=.4,
                    ax=ax[idx//4, idx%4], zorder=2)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s.legend();
    s.grid(True, axis='y', alpha=.4, zorder=0)

fig.tight_layout();


# ### Выводы
# **Время начала сессии с учетом дня недели**  
# Статистика по неделе с учетом времени начала сессии показывает, что для каждого пользователя характерно свое время выхода в сеть. Значит, такие признаки могут оказаться полезными при идентификации пользователя. Так как в данном случае мы будем учитывать некоторый паттерн поведения человека.
# 
# **Время дня, когда сессия началась**  
# Эта статистика показывает, в какое время дня для пользователя более характерно выходить в сеть. В целом в статистике отслеживается некоторый паттерн поведения, поэтому впоследствии такие признаки можно использовать при идентификации пользователя.

# **В конце сохраните в pickle-файлы только те признаки, которые, как Вы предполагаете, помогут идентифицировать пользователя более точно. Это касается и признаков, которые мы вместе создали в начале (*session_timespan, #unique_sites, start_hour, day_of_week*), и Ваших собственных. Можно создать все эти признаки не только для сессий из 10 сайтов, но и для других сочетаний параметров *session_length* и *window_size*.**

# In[37]:


selected_features_10users = new_features_10users[daytime_cols + ['dayofweek_hour']]
selected_features_150users = new_features_150users[daytime_cols + ['dayofweek_hour']]


# In[38]:


with open(os.path.join(PATH_TO_DATA, 
                       'selected_features_10users.pkl'), 'wb') as selected_features_10users_pkl:
    pickle.dump(selected_features_10users, selected_features_10users_pkl, 
                protocol=2)
with open(os.path.join(PATH_TO_DATA, 
                       'selected_features_150users.pkl'), 'wb') as selected_features_150users_pkl:
    pickle.dump(selected_features_150users, selected_features_150users_pkl, 
                protocol=2)


# ### Критерии оценки работы (только для Peer Review в специализации):
# - Верно ли отображена гистограмма session_timespan из п. 1? (max. 3 балла)
# - Верно ли отображена гистограмма #unique_sites из п. 2? (max. 3 балла)
# - Верно ли отображены гистограммы #unique_sites по каждому пользователю из п. 3? (max. 6 баллов)
# - Верно ли отображена гистограмма start_hour из п. 4? (max. 3 балла)
# - Верно ли отображены гистограммы start_hour по каждому пользователю из п. 5? (max. 6 баллов)
# - Верно ли отображена гистограмма day_of_week из п. 6? (max. 3 балла)
# - Верно ли отображены гистограммы day_of_week по каждому пользователю из п. 7? (max. 6 баллов)
# - Насколько сделанные выводы в п. 8 соответствуют построенным картинкам? (max. 6 баллов)
# - Верно ли отображен barplot для 10 популярных сайтов из п. 9? (max. 6 баллов)
# - Правильно ли посчитана медианная продолжительность сессий в п. 10? (max. 3 балла)
# - Правильно ли посчитан медианный день недели начала сессии в п. 11? (max. 3 балла)
# - Правильно ли посчитан медианный час начала сессии в п. 12? (max. 3 балла)
# - Правильно ли посчитано медианное значение числа уникальных сайтов в сессиях 150 пользователей п. 13? (max. 3 балла)
# - Есть ли оригинальные построенные признаки и картинки к ним? Оцените также и качество картинок. (max. 8 баллов)

# ## Пути улучшения
# Что еще можно добавить по 3 части проекта:
# - IPython-widgets, интерактив и анимация (стоящие статьи по этому ремеслу – [раз](https://habrahabr.ru/post/308162/) и  [два](https://habrahabr.ru/company/ods/blog/323210/))
# - можно попробовать изобразить исходные данные в некотором пространстве, например, Word2Vec, потом выделить главные компоненты или t-SNE  (только пользуйтесь эффективными реализациями типа [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE), не Sklearn) и раскрасить по целевому классу. Но нет гарантий, что получится что-то значимо отличающееся от каши
# 
# На следующей неделе мы наконец приступим к обучению моделей классификации. 
