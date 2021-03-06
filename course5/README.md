# Дополнительные материалы от авторов курса

- Заметка про 1-мерные и 2-мерные свертки и интерактивное демо, позволяющее попробовать разные фильтры: https://graphics.stanford.edu/courses/cs178/applets/convolution.html
- Более подробно про разнообразные виды блендинга изображений можно почитать на википедии: https://en.wikipedia.org/wiki/Blend_modes
- Подробная заметка про улучшение контраста с помощью эквилизации гистограммы: http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html
- Отличная книга про "низкоуровневое" компьютерное зрение и обработку изображений: **Вудс Р., Гонсалес Р.** *Цифровая обработка изображений*
- При решении задачи в первую очередь необходимо посмотреть, не решал ли её кто-то до этого, и нет ли готового решения. Для библиотеки caffe существует единый репозиторий моделей (зоопарк моделей, https://github.com/BVLC/caffe/wiki/Model-Zoo), можно попробовать искать решение в нём. В частности, там выложены несколько моделей, обученных на базе ImageNet, причём с лучшими результатами на мо- мент обучения. Таким образом, возможно, ничего не нужно придумывать, задача уже кем-то решена. Тогда достаточно взять готовую модель и использовать её на практике.
- VGG: https://github.com/ethereon/caffe-tensorflow

# Материалы для изучения нейронных сетей
*Материалы подготовил Василий Землянов, ментор специализации*

#### Neural Networks and Deep Learning

http://neuralnetworksanddeeplearning.com

Отличная короткая книга, всего 4 главы, каждая читается за вечер. Все очень наглядно разобрано, если засесть за упражнения можно значительно дольше залипнуть. Присутствует множество ссылок на научные статьи, в которые можно уйти с головой.

#### DMIA ( по нейронкам лекции 6-7)

Видео: https://www.youtube.com/channel/UCop3CelRVvrchG5lsPyxvHg/videos

Задачи: https://github.com/vkantor/MIPT_Data_Mining_In_Action_2016/tree/master/trends

Весь курс DMIA великолепен, лекции 6 и 7 посвящены нейронным сетям. В репозитории много интересных практических задач. На русском языке.

#### Легендарный CS231n

Видео: https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC

Материалы: http://cs231n.stanford.edu

Очень хороший курс по компьютерному зрению, в котором досконально разобран bleeding edge нейронных сетей. Автор курса, Andrej Karpathy, гуру сверточных нейронных сетей. у него офигенный блог и твиттер.

#### Второй курс Стенфорда по сетям, обработка естественных языков

Видео: https://www.youtube.com/playlist?list=PLmImxx8Char8dxWB9LRqdpCTmewaml96q

Материалы: http://cs224d.stanford.edu

Рекомендуют смотреть после Карпатого (cs231n)

#### DeepLearning book

http://www.deeplearningbook.org

https://github.com/HFTrader/DeepLearningBook (ссылка на PDF)

Топовая книга, на 11 из 10. Там и матан, и классическая машинка и сети и все что угодно. Очень подробно, очень матаново, очень круто.

#### BayArea School

Видео: https://www.youtube.com/channel/UCb7PaTJYueRh6Y5rQ7h3U3w/videos

Материалы: http://www.bayareadlschool.org/schedule

20 часов видео от ведущих специалистов, включая Andrej Karpathy и Andrew Ng

#### Лекции PhD из Франции

http://info.usherbrooke.ca/hlarochelle/neural_networks/description.html

Graduate курс по нейронным сетям.

#### Курс на степике

https://stepik.xn--org/course/--401-v3m35aab7ds6gax5c6a3a0x

Мне субъективно курс не очень нравится. Но он на русском языке и определенно стоит внимания.

#### Курс Хинтона

https://www.coursera.org/learn/neural-networks

Хинтон - человек который стоит у истоков современных нейронных сетей. Курс хороший, подробный, но, на мой взгляд, очень скучный.

#### Курс Udacity и Google по TensorFlow

https://www.udacity.com/course/deep-learning--ud730

Вводный практический курс. Простой, мало теории, много TensorFlow - одной из самых популярных библиотек.

#### Сборник научных статей по нейронным сетям

https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap

#### Репозитории для желающих почитать исходники:

https://github.com/tensorflow/tensorflow
https://github.com/Theano/Theano
https://github.com/Lasagne/Lasagne
https://github.com/fchollet/keras
https://github.com/dmlc/mxnet
https://github.com/torch