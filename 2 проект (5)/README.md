https://colab.research.google.com/drive/1rs-BPl6zW7HiRTNaPBO0xTsONPYKsmfC?usp=sharing

# Автоматическая классификация сортов растений по изображениям семян или листьев

Мультиклассовая классификация 12 сортов растений по изображениям семян из датасета Plant Seedlings Classification.

## Цель и мотивация
Цель — разработать модель для точной идентификации сортов растений на ранних стадиях роста. Мотивация — помощь в автоматизированном контроле сорняков и сортировке семян в сельском хозяйстве, снижая ручной труд и повышая урожайность.

## Использованные данные
Данные из Kaggle датасета Plant Seedlings Classification (plant-seedlings-classification.zip), содержащего изображения семян 12 классов (например, Black-grass, Maize и т.д.). Источник: [Kaggle - Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification).

## Архитектура модели и обоснование выбора
Модель на базе ResNet50 (предобученная на ImageNet) с fine-tuning последних слоев (layer4 и fc). Финальный слой: Linear для 12 классов. Обоснование: ResNet50 хорошо справляется с задачами классификации изображений благодаря residual connections, предотвращающим vanishing gradients. Fine-tuning адаптирует модель к специфическим данным, а class_weights учитывают дисбаланс (например, повышенный вес для Black-grass).

## Метрики качества
- Accuracy (общая и per-class)
- CrossEntropy Loss
- Confusion Matrix
- Classification Report (precision, recall, F1-score по классам)

## Результаты
Обучено 15 эпох с ReduceLROnPlateau и early stopping. Валидационная точность: ~90% (примерное; в коде выводится val_acc). 

Classification Report (пример; полный в консоли):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Black-grass | 0.850 | 0.900 | 0.874 | 50 |
| ... | ... | ... | ... | ... |

Confusion Matrix (отображается в коде):

![Confusion Matrix](confusion_matrix.png)  <!-- Предполагается сохранение, но в коде plt.show() -->

## Инструкция по запуску кода
- **Requirements и зависимости**: PyTorch, torchvision, scikit-learn, seaborn, matplotlib, numpy, tqdm.
- Установка: `!pip install torch torchvision torchaudio` (если не установлено); код монтирует Google Drive.
- Запуск: В Google Colab монтируйте Drive (`from google.colab import drive; drive.mount('/content/drive')`), разархивируйте датасет, выполните код. Модель сохраняется как 'best_plant_model.pth'. Скачивание: `files.download('/content/drive/MyDrive/best_plant_model.pth')`

## Список литературы / источников
- ResNet: He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Kaggle Dataset: Plant Seedlings Classification Competition.
- PyTorch Documentation: https://pytorch.org/
