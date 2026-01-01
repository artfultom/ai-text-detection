# План работы и обзор литературы

## Общая информация

Тема диплома: **Разработка системы для распознавания сгенерированных искусственным интеллектом текстов в академическом письме.**

Тема диплома (на английском): **Developing a System to Detect Machine-Generated Text in Academic Writing.**

Автор: Васильев Андрей Александрович.

Научный руководитель: Кантонистова Елена Олеговна.

Соруководитель: Неволина Арина Станиславовна.

## План работы над ВКР

### Чекпоинт 1. Формирование плана работы и обзор литературы (01.12.2025)

_Примечание: Данный документ является результатом первого чекпоинта._

### Чекпоинт 2. Разработка baseline-решения либо аналога (20.01.2026)
1. Подготовка данных (12.01.2026).
2. Оценка их качества (12.01.2026).
3. Разработка baseline-решения (12.01.2026).
4. Обзор возможных методов решения (16.01.2026).

### Чекпоинт 3. Проведение основной работы по ВКР (10.03.2026)
1. Обоснование выбранного решения (23.01.2026).
2. Реализация выбранного решения (27.02.2026).
3. Улучшение выбранного решения (27.02.2026).
4. Анализ результатов (06.03.2026).
5. Разработка сервиса (20.03.2026).

### Обязательная предзащита (01.04.2026)

### Чекпоинт 4. Доработка ВКР по отзывам от руководителя и комиссии (10.05.2026)

### Защита ВКР (20.05.2026)

## Обзор литературы

### Генерация синтетических данных
1. Mihai Nadas, Laura Diosan, Andreea Tomescu.
   "Synthetic Data Generation Using Large Language Models: Advances in Text and Code".
   arXiv preprint, arXiv:2503.14023, version 2. 2025

_В статье описаны ключевые подходы - от простого prompt-based поколения до retrieval-augmented и методов с итеративным самоулучшением - которые позволяют создавать примеры для задач классификации, question answering, перевода кода, исправления багов и др._

2. Xu Tan, Tao Qin, Jiang Bian et al.
   "Regeneration Learning: A Learning Paradigm for Data Generation".
   arXiv preprint, arXiv:2301.08846, version 1. 2023

_Предлагает новый подход к генерации данных, в котором вместо прямой генерации результата модель сначала создаёт абстрактное промежуточное описание, а затем генерирует финальный пример на его основе._

3. Yukyung Lee, Soonwon Ka, Bokyung Son et al.
   "Navigating the Path of Writing: Outline-guided Text Generation with Large Language Models".
   arXiv preprint, arXiv:2404.13919, version 2. 2025

_Основная идея статьи - использовать явный план / структуру / "каркас" текста как вход: это помогает модели понимать, какую цель преследует текст, какую структуру должен иметь, какие темы/разделы включить._

4. Yukyung Lee, Soonwon Ka, Bokyung Son et al.
   "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models".
   arXiv preprint, arXiv:2305.04091 , version 3. 2023

_Улучшение zero-shot рассуждений LLM, разбивая задачу на два этапа: сначала модель создает план решения, а затем пошагово решает задачу, следуя этому плану._

5. Jiyuan Ren, Zhaocheng Du, Zhihao Wen et al.
   "Few-shot LLM Synthetic Data with Distribution Matching".
   arXiv preprint, arXiv:2502.08661, version 2. 2025

_Предлагает метод SynAlign, который позволяет генерировать синтетические данные с помощью LLM так, чтобы они статистически соответствовали реальным. Подход сочетает выбор информативных примеров, извлечение скрытых языковых атрибутов и последующее выравнивание распределений синтетики и реальных данных._

6. Zeyu Qin, Qingxiu Dong, Xingxing Zhang et al.
   Scaling Laws of Synthetic Data for Language Models. 
   Proceedings of the Conference on Language Modeling (COLM). 2025.

_В статье предложен фреймворк SynthLLM, который превращает исходные корпусы текстов в масштабируемые, разнообразные и "высококачественные" синтетические датасеты, комбинируя концепции из разных документов._

### Оценка качества данных

1. German Gritsai, Anastasia Voznyuk, Andrey Grabovoy, Yury Chekhovich.
   "Are AI Detectors Good Enough? A Survey on Quality of Datasets With Machine-Generated Texts".
   arXiv preprint, arXiv:2410.14677, version 3. 2025

_Авторы анализируют существующие датасеты с текстами, сгенерированными ИИ, и указывают, что рекордные показатели точности детекторов часто связаны не с превосходством алгоритмов, а с низким качеством самих проверочных наборов.
В работе приводится систематический обзор датасетов из конкурсов и исследований по detection‑задачам, а также предложены методы для оценки качества этих датасетов - чтобы тестировать детекторы в более реалистичных и "жёстких" условиях._

2. Vijini Liyanage, Davide Buscaldi, Adeline Nazarenko.
   "A Benchmark Corpus for the Detection of Automatically Generated Text in Academic Publications".
   arXiv preprint, arXiv:2202.02013, version 2. 2022

_Авторы предлагают два набора данных - полностью синтетический и гибридный (где часть текста заменена машинно‑сгенерированной) - содержащих тексты, сгенерированные для академических публикаций, чтобы создать контрольный корпус для задач детекции текста, сгенерированного ИИ.
Они оценивают, насколько «человечными» выглядят такие тексты (по метрикам текучести и качества - BLEU, ROUGE), и демонстрируют, что многие из них достаточно похожи на реальные, что усложняет задачу различения._

### Выбор решения

1. Junchao Wu, Shu Yang, Runzhe Zhan et al.
   "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions".
   arXiv preprint, arXiv:2310.14724, version 3. 2024

_Обзорная статья с общими методами решения проблемы._

2. Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva et al.
   "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts".
   arXiv preprint, arXiv:2306.04723, version 2. 2023

_Авторы предлагают использовать внутреннюю размерность (intrinsic dimension) множества эмбеддингов текста как признак для детекции сгенерированного текста._

3. Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva et al.
   "Robust AI-Generated Text Detection by Restricted Embeddings".
   arXiv preprint, arXiv:2410.08113, version 1. 2024

_Авторы показывают, что удаление ("очистка") линейных подпространств (linear subspaces) в эмбеддинговом пространстве улучшает устойчивость (robustness) классификатора при детекции текста от разных LLM и в разных доменах._

4. F. Yin, J. Srinivasa, K. W. Chang.
   "Characterizing Truthfulness in Large Language Model Generations with Local Intrinsic Dimension".
   arXiv preprint, ArXiv:2402.18048, version 1. 2024

_В этой работе измеряется локальная внутренняя размерность (Local Intrinsic Dimension, LID) активаций модели для оценки "правдивости" (truthfulness) генерируемого текста._

5. Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani.
   "Short-PHD: Detecting Short LLM-Generated Text with Topological Data Analysis After Off-topic Content Insertion".
   arXiv preprint, arXiv:2504.02873, version 1. 2025

_Расширение идеи топологического анализа эмбеддингов: Persistent Homology Dimension (PHD) используется для детекции очень коротких текстов, с добавлением "вставки не по теме" (off-topic insertion) для стабильности оценки._
