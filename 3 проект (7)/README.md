https://colab.research.google.com/drive/1_NczZ0xMpCJ_6ZhJDTCBkdaNxpdzGouV?usp=sharing

# Автоматическая классификация сортов растений по изображениям семян или листьев

Мультиклассовая классификация 12 сортов растений по изображениям семян из датасета Plant Seedlings Classification.

## Цель и мотивация
Цель — разработать модель для точной идентификации сортов растений на ранних стадиях роста. Мотивация — помощь в автоматизированном контроле сорняков и сортировке семян в сельском хозяйстве, снижая ручной труд и повышая урожайность.

## Использованные данные
Данные из Kaggle датасета Plant Seedlings Classification (plant-seedlings-classification.zip), содержащего изображения семян 12 классов (например, Black-grass, Maize и т.д.). Источник: [Kaggle - Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification).

## Архитектура модели и обоснование выбора
Модель на базе ResNet50 (предобученная на ImageNet) с надстройкой последних слоев (layer4 и fc). Финальный слой заменён на линейный, выдающий 12 классов. ResNet50 отлично подходит для задач классификации изображений благодаря механизму остаточных связей (residual connections). Эти связи позволяют эффективно обучать очень глубокие сети, существенно снижая проблему исчезающих градиентов (vanishing gradients), которая обычно возникает в глубоких сетях. Надстройка адаптирует модель к специфическим данным, а веса классов учитывают дисбаланс (например, повышенный вес для Black-grass).

## Метрики качества
- Accuracy (общая и per-class)
- CrossEntropy Loss
- Confusion Matrix
- Classification Report (precision, recall, F1-score по классам)

## Результаты

<img width="1117" height="989" alt="confusion_matrix" src="https://github.com/user-attachments/assets/224694d7-8799-47b6-b217-d343d9b6bf6a" />

| Класс                        | Precision | Recall | F1-Score | Support |
|------------------------------|-----------|--------|----------|---------|
| Black-grass                  | 0.513     | 0.765  | 0.614    | 51      |
| Charlock                     | 0.973     | 1.000  | 0.986    | 71      |
| Cleavers                     | 0.967     | 0.935  | 0.951    | 62      |
| Common Chickweed             | 0.946     | 0.972  | 0.959    | 109     |
| Common wheat                 | 0.960     | 0.873  | 0.914    | 55      |
| Fat Hen                      | 0.980     | 0.960  | 0.970    | 100     |
| Loose Silky-bent             | 0.898     | 0.774  | 0.831    | 137     |
| Maize                        | 0.976     | 1.000  | 0.988    | 41      |
| Scentless Mayweed            | 0.907     | 0.942  | 0.925    | 104     |
| Shepherds Purse              | 0.911     | 0.854  | 0.882    | 48      |
| Small-flowered Cranesbill    | 0.990     | 0.981  | 0.986    | 105     |
| Sugar beet                   | 1.000     | 0.955  | 0.977    | 67      |
| **accuracy**                 |           |        | **0.917** | **950** |
| **macro avg**                | 0.918     | 0.918  | 0.915    | 950     |
| **weighted avg**             | 0.928     | 0.917  | 0.920    | 950     |

## Ключевые наблюдения

- **Модель очень сильная** на большинстве классов:  
  Charlock, Maize, Small-flowered Cranesbill, Fat Hen, Common Chickweed — почти безошибочное распознавание (recall ≥ 0.96).
- **Главная проблема** — сильная путаница между двумя похожими сорняками:  
  **Black-grass** и **Loose Silky-bent** — 39 ошибок из 43 общих (10 + 29 взаимных перепутываний).
- **Black-grass** чаще всего ошибочно относят к Loose Silky-bent (10 из 12 ошибок),  
  **Loose Silky-bent** — наоборот, в Black-grass (29 из 31 ошибок).
- Остальные ошибки редкие и единичные (по 1–5 на класс), модель уверенно различает 10 классов из 12.

## Инструкция по запуску кода
Открыть https://colab.research.google.com/drive/1_NczZ0xMpCJ_6ZhJDTCBkdaNxpdzGouV?usp=sharing.
Сделать копию себе на Гугл Диск.
Выполнить код поочередно во всех ячейках.
