# ML-marketing
ML-based Marketing, recommendation system for clustering people into subgroups to provide certain marketing strategy
*** Материалы использовать не старше 10 лет! ***
Overview на основные структуры (ссылка ПП - “medium”)
Recommender Systems: An Overview, Research Trends, and Future Directions (ссылка ПП - PDF)

QUESTIONS
1.  Какие типы архитектур чаще всего выбираются в бизнесе (Маркетинг в частности)? Примеры использования и описание

Существует три типа рекомендательных систем: контент-ориентированные (основаны на представлении предпочтений пользователей путем анализа содержимого рекомендательных элементов), коллаборативной фильтрации ( моделируют предпочтения, оценивая близость профилей пользователей), и гибридная система (объединяет в себе основные черты предыдущих двух, использует две модели сразу)

*Source: Types of Recommendation Systems & Their Use Cases 
Collaborative Filtering
The collaborative filtering method is based on gathering and analyzing data on user’s behavior. This includes the user’s online activities and predicting what they will like based on the similarity with other users.

For example, if user A likes Apple, Banana, and Mango while user B likes Apple, Banana, and Jackfruit, they have similar interests. So, it is highly likely that A would like Jackfruit and B would enjoy Mango. This is how collaborative filtering takes place.
Two kinds of collaborative filtering techniques used are:
User-User collaborative filtering
Item-Item collaborative filtering
One of the main advantages of this recommendation system is that it can recommend complex items precisely without understanding the object itself. There is no reliance on machine analyzable content. It requires no information about users or items and, so, they can be used in many situations.

But there is also a drawbag: as it only considers past interactions to make recommendations, collaborative filtering suffers from the “cold start problem”: it is impossible to recommend anything to new users or to recommend a new item to any users and many users or items have too few interactions to be efficiently handled. 

In memory based collaborative methods, no latent model is assumed. As no latent model is assumed, these methods have theoretically a low bias but a high variance.

In model based collaborative methods, some latent interaction model is assumed. New suggestions can then be done based on this model. As a (pretty free) model for user-item interactions is assumed, this method has theoretically a higher bias but a lower variance than methods assuming no latent model.
Content-Based Filtering
Content-based filtering methods are based on the description of a product and a profile of the user’s preferred choices. In this recommendation system, products are described using keywords, and a user profile is built to express the kind of item this user likes.

For instance, if a user likes to watch movies such as Iron Man, the recommender system recommends movies of the superhero genre or films describing Tony Stark.
The central assumption of content-based filtering is that you will also like a similar item if you like a particular item.

Advantage: content based methods suffer far less from the cold start problem than collaborative approaches: new users or items can be described by their characteristics (content) and so relevant suggestions can be done for these new entities.

Two disadvantages: at first, the systems act inaccurately and it takes more time to implement.
In content based methods some latent interaction model is also assumed.  Here, as for model based collaborative methods, a user-item interactions model is assumed. However, this model is more constrained and, so, the method tends to have the highest bias but the lowest variance.

In content based methods some latent interaction model is also assumed. Here, as for model based collaborative methods, a user-item interactions model is assumed. This model is more constrained and, so, the method tends to have the highest bias but the lowest variance.
Hybrid Recommendation Systems
In hybrid recommendation systems, products are recommended using both content-based and collaborative filtering simultaneously to suggest a broader range of products to customers. This recommendation system is up-and-coming and is said to provide more accurate recommendations than other recommender systems

Netflix is an excellent case in point of a hybrid recommendation system. It makes recommendations by juxtaposing users’ watching and searching habits and finding similar users on that platform. This way, Netflix uses collaborative filtering.

By recommending such shows/movies that share similar traits with those rated highly by the user, Netflix uses content-based filtering. They can also veto the common issues in recommendation systems, such as cold start and data insufficiency issues.
_______________________________________________________________________________
2. Как происходит обучение моделей, примеры обучения и примеры обучающей выборки? Source*
Generating Recommendation Matrix 
Assume that we have a simple user-item matrix, which shows the ratings of four users for five different movies. Let’s also assume that our active user has watched and rated three out of these five movies. Let’s find out which of the two movies that our active user hasn’t watched should be recommended to her.

The first step is to discover how similar the active user is to the other users. How do we do this? Well, this can be done through several different statistical and vectorial techniques such as distance or similarity measurements including Euclidean Distance, Pearson Correlation, Cosine Similarity, and so on. To calculate the level of similarity between two users, we use the three movies that both the users have rated in the past.

Regardless of what we use for similarity measurement, for example, the similarity could be 0.7, 0.9, and 0.4 between the active user and other users. These numbers represent similarity weights or proximity of the active user to other users in the dataset. The next step is to create a weighted rating matrix. We just calculated the similarity of users to our active user in Fig 2; now, we can use it to calculate the possible opinion of the active user about our two target movies. This is achieved by multiplying the similarity weights to the user ratings.

It results in a weighted ratings matrix, which represents the user’s neighbors’ opinions about our two candidate movies for recommendation. In fact, it incorporates the behavior of other users and gives more weight to the ratings of those users who are more similar to the active user.

Now, we can generate the recommendation matrix by aggregating all of the weighted rates. However, as three users rated the first potential movie and two users rated the second movie, we have to normalize the weighted rating values. We do this by dividing the sum of weighted ratings by the sum of the similarity index for users.

In the user-based approach, the recommendation is based on users of the same neighborhood with whom he or she shares common preferences. For example, as User 1 and User 3 both liked Item 3 and Item 4, we consider them as similar — or neighbor users — and recommend Item 1 which is positively rated by User 1 to User 3.

In the item-based approach, similar items build neighborhoods on the behavior of users (not based on their contents!). For example, Item 1 and Item 3 are considered neighbors as they were positively rated by both User 1 and User 2. So, Item 1 can be recommended to User 3 as he or she has already shown interest in Item 3. Therefore, the recommendations here are based on the items in the neighborhood that a user might prefer.

_______________________________________________________________________________
Implementation

The complete code is available as a Jupyter Notebook on GitHub

Loading Data
Data Inspection and Preparation
Data Preprocessing
Model Training
Top n-recommendations

________________________________________________________________________________



3. Наиболее популярные дата-сеты для проверки качества и обучения моделей, особенности алгоритмов обучения.

Recommender Systems and Personalization Datasets - Goodreads Book Reviews, Clothing Fit Data, Amazon Product Reviews
10 Open-Source Datasets One Must Know To Build Recommender Systems - MovieLens 25M Dataset, 
Common Datasets Benchmark for Recommendation System
Netflix Prize - многовариантный датасет временных рядов, который использовался в конкурсе Netflix Prize с рейтингами примерно 100 миллионов фильмов. В наборе данных более 480000 пользователей, каждый из которых промаркирован уникальным целочисленным идентификатором. С помощью этого датасета можно предсказать недостающие записи в матрице рейтинга пользователей фильма. 
Bookcrossing – датасет с рейтингами около 300 тысяч миллионов книг и обезличенными демографическими данными о более 250 тысячах их читателей. 
Google Search Dataset 
Kaggle
________________________________________________________________________________


4. Обзор на исследования и готовые рекомендательные системы, выявления плюсов и минусов с описанием.
-Что делали?
-Что нового сделали?
-Как сделали?
-Кто и как критиковал?
-Уместна ли критика?

________________________________________________________________________________
5* Доп: поискать применения МЛ в маркетинге с примерами и описанием
13 Examples of Machine Learning for Marketing (ссылка ПП)
Beginner Tutorial: Recommender Systems in Python
How to Build Simple Recommender Systems in Python
________________________________________________________________________________


USEFUL LINKS
Учебные работы (оформленные)
Диссертация какого-то крутого магистра белоруса тут 1 глава: теор основы построения рекомендательных систем

ВКР студента МФТИ: Построение рекомендательной системы, основанной на обучении с подкреплением

Научные статьи
Рекомендательные системы: идеи, подходы, задачи
Recommendation Systems — Models and Evaluation 
Yandex DataLens - их телеграмм, тут есть много полезных ссылок: обучение, обсуждение, датасеты, готовые проекты и тд
Design and implementation of a recommender system as a module for Liferay portal

Read Later
Какая-то тема с ашаном и bigdata
Introduction to recommender systems | by Baptiste Rocca | Towards Data Science
Где найти обучающие датасеты для рекомендательных систем.
 
________________________________________________________________________________

ADDITIONAL INFORMATION

Постановка задачи рекомендации
Задача: индивидуальный подбор наиболее релевантных предложений для клиента для достижения каких-либо бизнес-показателей

Актуальность: задачи формирования
Рекомендаций активно реализуются в каждой отрасли работы с клиентами (В2С): ритейл, финансовый сектор, телеком и пр.

Решение: экосистема рекомендаций начинается не с ранжирования лент с использованием сложных коллаборативных алгоритмов, а с ассоциативных правил и моделей классификации

Задача рекомендательной системы – проинформировать пользователя о товаре, который ему может быть наиболее интересен в данный момент времени. Клиент получает информацию, а сервис зарабатывает на предоставлении качественных услуг. Услуги — это не обязательно прямые продажи предлагаемого товара. Сервис также может зарабатывать на комиссионных или просто увеличивать лояльность пользователей, которая потом выливается в рекламные и иные доходы.
*Source: Анатомия рекомендательных систем

В центре любой рекомендательной системы находится так называемая матрица предпочтений. Это матрица, по одной из осей которой отложены все клиенты сервиса (Users), а по другой – объекты рекомендации (Items). На пересечении некоторых пар (user, item) данная матрица заполнена оценками (Ratings) – это известный нам показатель заинтересованности пользователя в данном товаре, выраженный по заданной шкале (например от 1 до 5).

