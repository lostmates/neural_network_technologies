# Прогноз урожайности зерновых по спутниковым данным и метеоинформации

## Описание задачи
- **Задача**: Регрессия — предсказание урожайности (ц/га) по временным рядам
- **Модели**: LSTM, Transformer, CNN-LSTM
- **Данные**: Спутниковые индексы (NDVI, EVI) + метеоданные (температура, осадки, солнечная радиация)
- **Регионы**: Краснодарский край, Ростовская область, Ставропольский край, Саратовская область, Алтайский край
- **Период**: 2010-2023 гг.

## 1. Импорт библиотек


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

    TensorFlow version: 2.20.0
    GPU available: []
    

## 2. Загрузка и анализ данных


```python
# Загрузка данных
data_path = '../data/raw/russia_crop_yield_dataset_2010_2023.csv'
df = pd.read_csv(data_path)

# Основная информация о данных
print("Форма датасета:", df.shape)
print("\nКолонки:", df.columns.tolist())
print("\nТипы данных:")
print(df.dtypes)
print("\nПропущенные значения:")
print(df.isnull().sum())
print("\nПервые 5 строк:")
df.head()
```

    Форма датасета: (840, 9)
    
    Колонки: ['region', 'year', 'month', 'NDVI', 'EVI', 'T2M_mean', 'PRECTOT_mm', 'SOLAR_RAD_MJ', 'yield_centners_per_ha']
    
    Типы данных:
    region                       str
    year                       int64
    month                      int64
    NDVI                     float64
    EVI                      float64
    T2M_mean                 float64
    PRECTOT_mm               float64
    SOLAR_RAD_MJ             float64
    yield_centners_per_ha    float64
    dtype: object
    
    Пропущенные значения:
    region                   0
    year                     0
    month                    0
    NDVI                     0
    EVI                      0
    T2M_mean                 0
    PRECTOT_mm               0
    SOLAR_RAD_MJ             0
    yield_centners_per_ha    0
    dtype: int64
    
    Первые 5 строк:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>year</th>
      <th>month</th>
      <th>NDVI</th>
      <th>EVI</th>
      <th>T2M_mean</th>
      <th>PRECTOT_mm</th>
      <th>SOLAR_RAD_MJ</th>
      <th>yield_centners_per_ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Krasnodar Krai</td>
      <td>2010</td>
      <td>1</td>
      <td>0.215</td>
      <td>0.178</td>
      <td>-3.92</td>
      <td>20.38</td>
      <td>5.47</td>
      <td>37.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Krasnodar Krai</td>
      <td>2010</td>
      <td>2</td>
      <td>0.205</td>
      <td>0.145</td>
      <td>-7.74</td>
      <td>32.83</td>
      <td>8.31</td>
      <td>37.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Krasnodar Krai</td>
      <td>2010</td>
      <td>3</td>
      <td>0.222</td>
      <td>0.169</td>
      <td>0.85</td>
      <td>32.75</td>
      <td>8.13</td>
      <td>37.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Krasnodar Krai</td>
      <td>2010</td>
      <td>4</td>
      <td>0.343</td>
      <td>0.287</td>
      <td>13.31</td>
      <td>48.78</td>
      <td>14.57</td>
      <td>39.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Krasnodar Krai</td>
      <td>2010</td>
      <td>5</td>
      <td>0.488</td>
      <td>0.378</td>
      <td>15.92</td>
      <td>57.65</td>
      <td>18.70</td>
      <td>42.66</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Статистическое описание данных
print("Статистическое описание числовых признаков:")
df.describe()
```

    Статистическое описание числовых признаков:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>NDVI</th>
      <th>EVI</th>
      <th>T2M_mean</th>
      <th>PRECTOT_mm</th>
      <th>SOLAR_RAD_MJ</th>
      <th>yield_centners_per_ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
      <td>840.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2016.500000</td>
      <td>6.500000</td>
      <td>0.409299</td>
      <td>0.327707</td>
      <td>7.851964</td>
      <td>38.593667</td>
      <td>15.105143</td>
      <td>39.340524</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.033531</td>
      <td>3.454109</td>
      <td>0.177399</td>
      <td>0.143533</td>
      <td>8.958069</td>
      <td>14.579416</td>
      <td>6.644658</td>
      <td>7.900594</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2010.000000</td>
      <td>1.000000</td>
      <td>0.142000</td>
      <td>0.090000</td>
      <td>-10.340000</td>
      <td>8.450000</td>
      <td>3.690000</td>
      <td>23.140000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2013.000000</td>
      <td>3.750000</td>
      <td>0.246000</td>
      <td>0.199750</td>
      <td>-0.030000</td>
      <td>26.010000</td>
      <td>8.935000</td>
      <td>32.915000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2016.500000</td>
      <td>6.500000</td>
      <td>0.367500</td>
      <td>0.292000</td>
      <td>8.190000</td>
      <td>36.580000</td>
      <td>14.485000</td>
      <td>39.090000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2020.000000</td>
      <td>9.250000</td>
      <td>0.577250</td>
      <td>0.456750</td>
      <td>16.172500</td>
      <td>51.592500</td>
      <td>20.915000</td>
      <td>45.420000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023.000000</td>
      <td>12.000000</td>
      <td>0.767000</td>
      <td>0.629000</td>
      <td>24.750000</td>
      <td>70.090000</td>
      <td>27.330000</td>
      <td>58.810000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Анализ регионов
print("Уникальные регионы:")
regions = df['region'].unique()
print(regions)
print(f"\nКоличество регионов: {len(regions)}")

# Количество записей по регионам
print("\nКоличество записей по регионам:")
df['region'].value_counts()
```

    Уникальные регионы:
    <StringArray>
    ['Krasnodar Krai',  'Rostov Oblast', 'Stavropol Krai', 'Saratov Oblast',
         'Altai Krai']
    Length: 5, dtype: str
    
    Количество регионов: 5
    
    Количество записей по регионам:
    




    region
    Krasnodar Krai    168
    Rostov Oblast     168
    Stavropol Krai    168
    Saratov Oblast    168
    Altai Krai        168
    Name: count, dtype: int64



## 3. Визуализация данных


```python
# Визуализация распределения урожайности по регионам
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, region in enumerate(regions):
    region_data = df[df['region'] == region]
    axes[i].hist(region_data['yield_centners_per_ha'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{region}')
    axes[i].set_xlabel('Урожайность (ц/га)')
    axes[i].set_ylabel('Частота')

# Удаляем лишний subplot
axes[-1].remove()

plt.suptitle('Распределение урожайности по регионам', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    



```python
# Динамика урожайности по годам для каждого региона
fig, ax = plt.subplots(figsize=(14, 7))

for region in regions:
    region_data = df[df['region'] == region]
    yearly_yield = region_data.groupby('year')['yield_centners_per_ha'].mean()
    ax.plot(yearly_yield.index, yearly_yield.values, marker='o', linewidth=2, label=region)

ax.set_xlabel('Год', fontsize=12)
ax.set_ylabel('Средняя урожайность (ц/га)', fontsize=12)
ax.set_title('Динамика урожайности по годам', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    



```python
# Сезонная динамика NDVI и EVI
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# NDVI по месяцам
for region in regions:
    region_data = df[df['region'] == region]
    monthly_ndvi = region_data.groupby('month')['NDVI'].mean()
    axes[0].plot(monthly_ndvi.index, monthly_ndvi.values, marker='o', linewidth=2, label=region)

axes[0].set_xlabel('Месяц', fontsize=12)
axes[0].set_ylabel('NDVI', fontsize=12)
axes[0].set_title('Сезонная динамика NDVI', fontsize=14, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(1, 13))

# EVI по месяцам
for region in regions:
    region_data = df[df['region'] == region]
    monthly_evi = region_data.groupby('month')['EVI'].mean()
    axes[1].plot(monthly_evi.index, monthly_evi.values, marker='o', linewidth=2, label=region)

axes[1].set_xlabel('Месяц', fontsize=12)
axes[1].set_ylabel('EVI', fontsize=12)
axes[1].set_title('Сезонная динамика EVI', fontsize=14, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, 13))

plt.tight_layout()
plt.show()
```


    
![png](output_10_0.png)
    



```python
# Корреляционная матрица
numeric_cols = ['NDVI', 'EVI', 'T2M_mean', 'PRECTOT_mm', 'SOLAR_RAD_MJ', 'yield_centners_per_ha']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    


## 4. Предобработка данных


```python
# Кодирование регионов
region_mapping = {region: idx for idx, region in enumerate(regions)}
df['region_encoded'] = df['region'].map(region_mapping)

print("Кодирование регионов:")
for region, code in region_mapping.items():
    print(f"{region}: {code}")
```

    Кодирование регионов:
    Krasnodar Krai: 0
    Rostov Oblast: 1
    Stavropol Krai: 2
    Saratov Oblast: 3
    Altai Krai: 4
    


```python
# Подготовка признаков
feature_cols = ['NDVI', 'EVI', 'T2M_mean', 'PRECTOT_mm', 'SOLAR_RAD_MJ', 'region_encoded']
target_col = 'yield_centners_per_ha'

X = df[feature_cols].values
y = df[target_col].values

print(f"Размерность признаков: {X.shape}")
print(f"Размерность целевой переменной: {y.shape}")
```

    Размерность признаков: (840, 6)
    Размерность целевой переменной: (840,)
    


```python
# Нормализация данных
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("Нормализация выполнена")
print(f"Признаки - Mean: {X_scaled.mean(axis=0).round(4)}, Std: {X_scaled.std(axis=0).round(4)}")
print(f"Целевая переменная - Min: {y_scaled.min():.4f}, Max: {y_scaled.max():.4f}")
```

    Нормализация выполнена
    Признаки - Mean: [-0. -0.  0.  0.  0.  0.], Std: [1. 1. 1. 1. 1. 1.]
    Целевая переменная - Min: 0.0000, Max: 1.0000
    


```python
# Создание временных последовательностей
def create_sequences(data, target, sequence_length=12):
    """
    Создание последовательностей для временных рядов
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        X_seq.append(data[i:(i + sequence_length)])
        y_seq.append(target[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

SEQUENCE_LENGTH = 12  # 12 месяцев

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

print(f"Размерность последовательностей X: {X_seq.shape}")
print(f"Размерность целевой переменной y: {y_seq.shape}")
```

    Размерность последовательностей X: (828, 12, 6)
    Размерность целевой переменной y: (828,)
    


```python
# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
```

    Train set: (662, 12, 6), (662,)
    Test set: (166, 12, 6), (166,)
    

## 5. Определение моделей


```python
# LSTM Model
def build_lstm_model(sequence_length, n_features):
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, 
                         input_shape=(sequence_length, n_features)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
```


```python
# CNN-LSTM Model
def build_cnn_lstm_model(sequence_length, n_features):
    model = keras.Sequential([
        keras.layers.Conv1D(64, 3, activation='relu', padding='same',
                           input_shape=(sequence_length, n_features)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
```


```python
# Transformer Block
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer Model
def build_transformer_model(sequence_length, n_features):
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_blocks = 2
    dropout_rate = 0.1
    
    inputs = keras.layers.Input(shape=(sequence_length, n_features))
    x = keras.layers.Dense(embed_dim)(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_embedding = keras.layers.Embedding(sequence_length, embed_dim)(positions)
    x = x + pos_embedding
    
    # Transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
```

## 6. Обучение моделей


```python
# Параметры обучения
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

n_features = X_train.shape[2]
print(f"Количество признаков: {n_features}")
print(f"Длина последовательности: {SEQUENCE_LENGTH}")
```

    Количество признаков: 6
    Длина последовательности: 12
    

### 6.1 LSTM Model


```python
# Build and train LSTM
lstm_model = build_lstm_model(SEQUENCE_LENGTH, n_features)
lstm_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)             │          <span style="color: #00af00; text-decoration-color: #00af00">69,120</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)             │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │          <span style="color: #00af00; text-decoration-color: #00af00">49,408</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">120,641</span> (471.25 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">120,641</span> (471.25 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

# Train LSTM
history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

    Epoch 1/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 45ms/step - loss: 0.0815 - mae: 0.2185 - val_loss: 0.1248 - val_mae: 0.3427 - learning_rate: 0.0010
    Epoch 2/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0329 - mae: 0.1434 - val_loss: 0.0527 - val_mae: 0.1940 - learning_rate: 0.0010
    Epoch 3/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 33ms/step - loss: 0.0234 - mae: 0.1200 - val_loss: 0.0146 - val_mae: 0.0935 - learning_rate: 0.0010
    Epoch 4/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.0218 - mae: 0.1130 - val_loss: 0.0154 - val_mae: 0.1019 - learning_rate: 0.0010
    Epoch 5/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0219 - mae: 0.1168 - val_loss: 0.0178 - val_mae: 0.1016 - learning_rate: 0.0010
    Epoch 6/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.0208 - mae: 0.1135 - val_loss: 0.0133 - val_mae: 0.0920 - learning_rate: 0.0010
    Epoch 7/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0189 - mae: 0.1089 - val_loss: 0.0127 - val_mae: 0.0845 - learning_rate: 0.0010
    Epoch 8/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.0190 - mae: 0.1079 - val_loss: 0.0173 - val_mae: 0.1113 - learning_rate: 0.0010
    Epoch 9/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.0194 - mae: 0.1081 - val_loss: 0.0150 - val_mae: 0.0918 - learning_rate: 0.0010
    Epoch 10/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.0180 - mae: 0.1072 - val_loss: 0.0190 - val_mae: 0.1142 - learning_rate: 0.0010
    Epoch 11/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.0162 - mae: 0.1000 - val_loss: 0.0256 - val_mae: 0.1198 - learning_rate: 0.0010
    Epoch 12/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0168 - mae: 0.1027 - val_loss: 0.0193 - val_mae: 0.1106 - learning_rate: 0.0010
    Epoch 13/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0168 - mae: 0.1025 - val_loss: 0.0206 - val_mae: 0.1170 - learning_rate: 5.0000e-04
    Epoch 14/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.0173 - mae: 0.1014 - val_loss: 0.0176 - val_mae: 0.0961 - learning_rate: 5.0000e-04
    Epoch 15/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.0156 - mae: 0.0982 - val_loss: 0.0186 - val_mae: 0.1112 - learning_rate: 5.0000e-04
    Epoch 16/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.0148 - mae: 0.0950 - val_loss: 0.0166 - val_mae: 0.0942 - learning_rate: 5.0000e-04
    Epoch 17/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.0151 - mae: 0.0979 - val_loss: 0.0183 - val_mae: 0.0996 - learning_rate: 5.0000e-04
    Epoch 18/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0148 - mae: 0.0934 - val_loss: 0.0178 - val_mae: 0.1027 - learning_rate: 2.5000e-04
    Epoch 19/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.0138 - mae: 0.0940 - val_loss: 0.0177 - val_mae: 0.0996 - learning_rate: 2.5000e-04
    Epoch 20/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.0134 - mae: 0.0920 - val_loss: 0.0187 - val_mae: 0.0992 - learning_rate: 2.5000e-04
    Epoch 21/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.0131 - mae: 0.0912 - val_loss: 0.0220 - val_mae: 0.1083 - learning_rate: 2.5000e-04
    Epoch 22/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.0134 - mae: 0.0908 - val_loss: 0.0189 - val_mae: 0.1041 - learning_rate: 2.5000e-04
    


```python
# Plot LSTM training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_lstm.history['loss'], label='Train Loss')
axes[0].plot(history_lstm.history['val_loss'], label='Validation Loss')
axes[0].set_title('LSTM: Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_lstm.history['mae'], label='Train MAE')
axes[1].plot(history_lstm.history['val_mae'], label='Validation MAE')
axes[1].set_title('LSTM: MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('LSTM Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_27_0.png)
    


### 6.2 CNN-LSTM Model


```python
# Build and train CNN-LSTM
cnn_lstm_model = build_cnn_lstm_model(SEQUENCE_LENGTH, n_features)
cnn_lstm_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">1,216</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │           <span style="color: #00af00; text-decoration-color: #00af00">6,176</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │             <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │          <span style="color: #00af00; text-decoration-color: #00af00">24,832</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,721</span> (135.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,529</span> (134.88 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">192</span> (768.00 B)
</pre>




```python
# Train CNN-LSTM
history_cnn_lstm = cnn_lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

    Epoch 1/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 49ms/step - loss: 0.0888 - mae: 0.2287 - val_loss: 0.3690 - val_mae: 0.5924 - learning_rate: 0.0010
    Epoch 2/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0403 - mae: 0.1588 - val_loss: 0.3868 - val_mae: 0.6076 - learning_rate: 0.0010
    Epoch 3/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.0323 - mae: 0.1404 - val_loss: 0.4311 - val_mae: 0.6439 - learning_rate: 0.0010
    Epoch 4/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0257 - mae: 0.1268 - val_loss: 0.4370 - val_mae: 0.6488 - learning_rate: 0.0010
    Epoch 5/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0219 - mae: 0.1178 - val_loss: 0.4506 - val_mae: 0.6593 - learning_rate: 0.0010
    Epoch 6/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0234 - mae: 0.1193 - val_loss: 0.4532 - val_mae: 0.6608 - learning_rate: 5.0000e-04
    Epoch 7/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step - loss: 0.0231 - mae: 0.1153 - val_loss: 0.4519 - val_mae: 0.6592 - learning_rate: 5.0000e-04
    Epoch 8/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0189 - mae: 0.1079 - val_loss: 0.4388 - val_mae: 0.6480 - learning_rate: 5.0000e-04
    Epoch 9/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0204 - mae: 0.1118 - val_loss: 0.4350 - val_mae: 0.6456 - learning_rate: 5.0000e-04
    Epoch 10/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step - loss: 0.0197 - mae: 0.1096 - val_loss: 0.4135 - val_mae: 0.6288 - learning_rate: 5.0000e-04
    Epoch 11/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0168 - mae: 0.1009 - val_loss: 0.3813 - val_mae: 0.6025 - learning_rate: 2.5000e-04
    Epoch 12/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.0169 - mae: 0.1041 - val_loss: 0.3587 - val_mae: 0.5835 - learning_rate: 2.5000e-04
    Epoch 13/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - loss: 0.0176 - mae: 0.1038 - val_loss: 0.3359 - val_mae: 0.5642 - learning_rate: 2.5000e-04
    Epoch 14/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step - loss: 0.0163 - mae: 0.1013 - val_loss: 0.3206 - val_mae: 0.5516 - learning_rate: 2.5000e-04
    Epoch 15/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step - loss: 0.0164 - mae: 0.0984 - val_loss: 0.3013 - val_mae: 0.5340 - learning_rate: 2.5000e-04
    


```python
# Plot CNN-LSTM training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_cnn_lstm.history['loss'], label='Train Loss')
axes[0].plot(history_cnn_lstm.history['val_loss'], label='Validation Loss')
axes[0].set_title('CNN-LSTM: Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_cnn_lstm.history['mae'], label='Train MAE')
axes[1].plot(history_cnn_lstm.history['val_mae'], label='Validation MAE')
axes[1].set_title('CNN-LSTM: MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('CNN-LSTM Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_31_0.png)
    


### 6.3 Transformer Model


```python
# Build and train Transformer
transformer_model = build_transformer_model(SEQUENCE_LENGTH, n_features)
transformer_model.summary()
```

    WARNING:tensorflow:From D:\Uni\магистратура\3 семестр\Аврахам\crop_yield_prediction\venv\Lib\site-packages\keras\src\backend\tensorflow\core.py:232: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_4"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)                            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerBlock</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">83,200</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ transformer_block_1                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">83,200</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerBlock</span>)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling1d             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">168,961</span> (660.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">168,961</span> (660.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Train Transformer
history_transformer = transformer_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

    Epoch 1/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 57ms/step - loss: 0.1418 - mae: 0.2891 - val_loss: 0.0269 - val_mae: 0.1329 - learning_rate: 0.0010
    Epoch 2/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.0444 - mae: 0.1675 - val_loss: 0.0218 - val_mae: 0.1139 - learning_rate: 0.0010
    Epoch 3/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0343 - mae: 0.1491 - val_loss: 0.0376 - val_mae: 0.1507 - learning_rate: 0.0010
    Epoch 4/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0332 - mae: 0.1457 - val_loss: 0.0237 - val_mae: 0.1225 - learning_rate: 0.0010
    Epoch 5/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.0323 - mae: 0.1438 - val_loss: 0.0393 - val_mae: 0.1636 - learning_rate: 0.0010
    Epoch 6/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.0292 - mae: 0.1372 - val_loss: 0.0313 - val_mae: 0.1430 - learning_rate: 5.0000e-04
    Epoch 7/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.0285 - mae: 0.1349 - val_loss: 0.0245 - val_mae: 0.1239 - learning_rate: 5.0000e-04
    Epoch 8/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0271 - mae: 0.1326 - val_loss: 0.0261 - val_mae: 0.1307 - learning_rate: 5.0000e-04
    Epoch 9/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.0237 - mae: 0.1243 - val_loss: 0.0250 - val_mae: 0.1260 - learning_rate: 5.0000e-04
    Epoch 10/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.0256 - mae: 0.1298 - val_loss: 0.0232 - val_mae: 0.1205 - learning_rate: 5.0000e-04
    Epoch 11/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.0233 - mae: 0.1204 - val_loss: 0.0254 - val_mae: 0.1273 - learning_rate: 2.5000e-04
    Epoch 12/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0229 - mae: 0.1230 - val_loss: 0.0254 - val_mae: 0.1270 - learning_rate: 2.5000e-04
    Epoch 13/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0230 - mae: 0.1212 - val_loss: 0.0265 - val_mae: 0.1286 - learning_rate: 2.5000e-04
    Epoch 14/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0246 - mae: 0.1270 - val_loss: 0.0272 - val_mae: 0.1332 - learning_rate: 2.5000e-04
    Epoch 15/100
    [1m17/17[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.0233 - mae: 0.1217 - val_loss: 0.0235 - val_mae: 0.1210 - learning_rate: 2.5000e-04
    


```python
# Plot Transformer training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_transformer.history['loss'], label='Train Loss')
axes[0].plot(history_transformer.history['val_loss'], label='Validation Loss')
axes[0].set_title('Transformer: Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_transformer.history['mae'], label='Train MAE')
axes[1].plot(history_transformer.history['val_mae'], label='Validation MAE')
axes[1].set_title('Transformer: MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Transformer Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_35_0.png)
    


## 7. Оценка моделей


```python
# Предсказания на тестовом наборе
y_pred_lstm = lstm_model.predict(X_test)
y_pred_cnn_lstm = cnn_lstm_model.predict(X_test)
y_pred_transformer = transformer_model.predict(X_test)

# Обратное преобразование к исходному масштабу
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_lstm_original = scaler_y.inverse_transform(y_pred_lstm).flatten()
y_pred_cnn_lstm_original = scaler_y.inverse_transform(y_pred_cnn_lstm).flatten()
y_pred_transformer_original = scaler_y.inverse_transform(y_pred_transformer).flatten()
```

    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 53ms/step
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
    WARNING:tensorflow:5 out of the last 13 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000000511BA980> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
    


```python
# Функция для расчета метрик
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Расчет метрик для всех моделей
metrics = {}

metrics['LSTM'] = calculate_metrics(y_test_original, y_pred_lstm_original)
metrics['CNN-LSTM'] = calculate_metrics(y_test_original, y_pred_cnn_lstm_original)
metrics['Transformer'] = calculate_metrics(y_test_original, y_pred_transformer_original)

# Вывод результатов
print("Результаты оценки моделей:")
print("=" * 50)
print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15}")
print("-" * 50)
for model_name, (mae, rmse) in metrics.items():
    print(f"{model_name:<15} {mae:<15.4f} {rmse:<15.4f}")
```

    Результаты оценки моделей:
    ==================================================
    Model           MAE             RMSE           
    --------------------------------------------------
    LSTM            27.3410         27.5692        
    CNN-LSTM        8.4640          9.6616         
    Transformer     21.4480         22.0274        
    


```python
# Визуализация сравнения метрик
models = list(metrics.keys())
mae_values = [metrics[m][0] for m in models]
rmse_values = [metrics[m][1] for m in models]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
bars1 = axes[0].bar(models, mae_values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[0].set_title('MAE Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('MAE (ц/га)')
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, mae_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# RMSE comparison
bars2 = axes[1].bar(models, rmse_values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RMSE (ц/га)')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, rmse_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```


    
![png](output_39_0.png)
    


## 8. Визуализация прогнозов по регионам


```python
# Функция для получения предсказаний с привязкой к регионам
def get_predictions_by_region(df, X_seq, y_pred, sequence_length=12):
    """
    Привязка предсказаний к регионам
    """
    # Создаем DataFrame для предсказаний
    pred_df = df.iloc[sequence_length:].copy().reset_index(drop=True)
    pred_df = pred_df.iloc[:len(y_pred)].copy()
    pred_df['predicted_yield'] = y_pred
    return pred_df

# Получаем предсказания для лучшей модели (выбираем по минимальному MAE)
best_model_name = min(metrics, key=lambda x: metrics[x][0])
print(f"Лучшая модель: {best_model_name}")

if best_model_name == 'LSTM':
    best_predictions = y_pred_lstm_original
elif best_model_name == 'CNN-LSTM':
    best_predictions = y_pred_cnn_lstm_original
else:
    best_predictions = y_pred_transformer_original

# Получаем тестовые данные с регионами
test_df = df.iloc[SEQUENCE_LENGTH:].copy().reset_index(drop=True)
test_indices = range(len(X_train), len(X_train) + len(X_test))
test_df = test_df.iloc[test_indices].copy()
test_df = test_df.iloc[:len(y_test_original)].copy()
test_df['actual_yield'] = y_test_original
test_df['predicted_yield'] = best_predictions
```

    Лучшая модель: CNN-LSTM
    


```python
# Визуализация прогнозов по регионам
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, region in enumerate(regions):
    region_data = test_df[test_df['region'] == region]
    
    if len(region_data) > 0:
        axes[i].scatter(region_data['actual_yield'], region_data['predicted_yield'], 
                       alpha=0.6, s=50, color='blue')
        
        # Линия идеального прогноза
        min_val = min(region_data['actual_yield'].min(), region_data['predicted_yield'].min())
        max_val = max(region_data['actual_yield'].max(), region_data['predicted_yield'].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальный прогноз')
        
        axes[i].set_xlabel('Фактическая урожайность (ц/га)')
        axes[i].set_ylabel('Прогнозируемая урожайность (ц/га)')
        axes[i].set_title(f'{region}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Добавляем R²
        from sklearn.metrics import r2_score
        r2 = r2_score(region_data['actual_yield'], region_data['predicted_yield'])
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Удаляем лишний subplot
axes[-1].remove()

plt.suptitle(f'Прогноз vs Факт ({best_model_name})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](output_42_0.png)
    



```python
# Временной ряд прогнозов по регионам
fig, axes = plt.subplots(len(regions), 1, figsize=(14, 16))

for i, region in enumerate(regions):
    region_data = test_df[test_df['region'] == region].copy()
    
    if len(region_data) > 0:
        region_data['date'] = pd.to_datetime(region_data[['year', 'month']].assign(day=1))
        region_data = region_data.sort_values('date')
        
        axes[i].plot(region_data['date'], region_data['actual_yield'], 
                    'o-', label='Факт', linewidth=2, markersize=4)
        axes[i].plot(region_data['date'], region_data['predicted_yield'], 
                    's--', label='Прогноз', linewidth=2, markersize=4)
        
        axes[i].set_title(f'{region}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Урожайность (ц/га)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Рассчитываем ошибку для региона
        mae_region = mean_absolute_error(region_data['actual_yield'], region_data['predicted_yield'])
        rmse_region = np.sqrt(mean_squared_error(region_data['actual_yield'], region_data['predicted_yield']))
        axes[i].text(0.02, 0.95, f'MAE: {mae_region:.2f}, RMSE: {rmse_region:.2f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

axes[-1].set_xlabel('Дата')
plt.suptitle(f'Временные ряды прогнозов ({best_model_name})', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
```


    
![png](output_43_0.png)
    



```python
# Распределение ошибок по регионам
test_df['error'] = test_df['actual_yield'] - test_df['predicted_yield']
test_df['abs_error'] = np.abs(test_df['error'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot ошибок по регионам
error_by_region = [test_df[test_df['region'] == r]['error'].values for r in regions]
bp = axes[0].boxplot(error_by_region, labels=[r.split()[0] for r in regions], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[0].set_title('Распределение ошибок по регионам', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ошибка (ц/га)')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='r', linestyle='--')

# Средняя абсолютная ошибка по регионам
mae_by_region = test_df.groupby('region')['abs_error'].mean().sort_values()
bars = axes[1].barh(range(len(mae_by_region)), mae_by_region.values, color='coral')
axes[1].set_yticks(range(len(mae_by_region)))
axes[1].set_yticklabels([r.split()[0] for r in mae_by_region.index])
axes[1].set_xlabel('MAE (ц/га)')
axes[1].set_title('Средняя абсолютная ошибка по регионам', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, mae_by_region.values)):
    axes[1].text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()
```


    
![png](output_44_0.png)
    


## 9. Сравнение всех моделей по регионам


```python
# Создаем DataFrame с предсказаниями всех моделей
comparison_df = test_df[['region', 'year', 'month', 'actual_yield']].copy()
comparison_df['LSTM'] = y_pred_lstm_original[:len(comparison_df)]
comparison_df['CNN-LSTM'] = y_pred_cnn_lstm_original[:len(comparison_df)]
comparison_df['Transformer'] = y_pred_transformer_original[:len(comparison_df)]

# Рассчитываем MAE по регионам для каждой модели
region_metrics = []
for region in regions:
    region_data = comparison_df[comparison_df['region'] == region]
    if len(region_data) > 0:
        for model_name in ['LSTM', 'CNN-LSTM', 'Transformer']:
            mae = mean_absolute_error(region_data['actual_yield'], region_data[model_name])
            rmse = np.sqrt(mean_squared_error(region_data['actual_yield'], region_data[model_name]))
            region_metrics.append({
                'region': region,
                'model': model_name,
                'MAE': mae,
                'RMSE': rmse
            })

region_metrics_df = pd.DataFrame(region_metrics)
print("Метрики по регионам:")
region_metrics_df.pivot(index='region', columns='model', values='MAE').round(2)
```

    Метрики по регионам:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>CNN-LSTM</th>
      <th>LSTM</th>
      <th>Transformer</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Altai Krai</th>
      <td>8.46</td>
      <td>27.34</td>
      <td>21.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Визуализация сравнения моделей по регионам
pivot_mae = region_metrics_df.pivot(index='region', columns='model', values='MAE')

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(pivot_mae.index))
width = 0.25

bars1 = ax.bar(x - width, pivot_mae['LSTM'], width, label='LSTM', color='#3498db')
bars2 = ax.bar(x, pivot_mae['CNN-LSTM'], width, label='CNN-LSTM', color='#2ecc71')
bars3 = ax.bar(x + width, pivot_mae['Transformer'], width, label='Transformer', color='#e74c3c')

ax.set_xlabel('Регион')
ax.set_ylabel('MAE (ц/га)')
ax.set_title('Сравнение моделей по регионам (MAE)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([r.split()[0] for r in pivot_mae.index], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```


    
![png](output_47_0.png)
    


## 10. Сохранение моделей


```python
import os

# Создаем директорию для моделей
models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)

# Сохраняем модели
lstm_model.save(f'{models_dir}/lstm_model.h5')
cnn_lstm_model.save(f'{models_dir}/cnn_lstm_model.h5')
transformer_model.save(f'{models_dir}/transformer_model.h5')

print("Модели сохранены:")
print(f"- {models_dir}/lstm_model.h5")
print(f"- {models_dir}/cnn_lstm_model.h5")
print(f"- {models_dir}/transformer_model.h5")
```

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    

    Модели сохранены:
    - ../models/lstm_model.h5
    - ../models/cnn_lstm_model.h5
    - ../models/transformer_model.h5
    


```python
# Сохранение скейлеров
import joblib

joblib.dump(scaler_X, f'{models_dir}/scaler_X.pkl')
joblib.dump(scaler_y, f'{models_dir}/scaler_y.pkl')

print("Скейлеры сохранены:")
print(f"- {models_dir}/scaler_X.pkl")
print(f"- {models_dir}/scaler_y.pkl")
```

    Скейлеры сохранены:
    - ../models/scaler_X.pkl
    - ../models/scaler_y.pkl
    

## 11. Выводы


```python
# Итоговая сводка
print("=" * 60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"\nЛучшая модель: {best_model_name}")
print(f"MAE: {metrics[best_model_name][0]:.4f} ц/га")
print(f"RMSE: {metrics[best_model_name][1]:.4f} ц/га")

print("\n" + "-" * 60)
print("Сравнение всех моделей:")
print("-" * 60)
for model_name, (mae, rmse) in metrics.items():
    print(f"{model_name:<15} MAE: {mae:.4f} ц/га, RMSE: {rmse:.4f} ц/га")

print("\n" + "=" * 60)
print("ОСОБЕННОСТИ ДАННЫХ:")
print("=" * 60)
print(f"• Регионы: {', '.join(regions)}")
print(f"• Период: {df['year'].min()}-{df['year'].max()}")
print(f"• Признаки: NDVI, EVI, температура, осадки, солнечная радиация")
print(f"• Целевая переменная: урожайность зерновых (ц/га)")

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ:")
print("=" * 60)
print("• Для улучшения точности можно добавить больше метеорологических признаков")
print("• Использовать более длинные временные последовательности")
print("• Применить ансамблирование моделей")
print("• Добавить данные о типах почв и сельскохозяйственных практиках")
```

    ============================================================
    ИТОГОВЫЕ РЕЗУЛЬТАТЫ
    ============================================================
    
    Лучшая модель: CNN-LSTM
    MAE: 8.4640 ц/га
    RMSE: 9.6616 ц/га
    
    ------------------------------------------------------------
    Сравнение всех моделей:
    ------------------------------------------------------------
    LSTM            MAE: 27.3410 ц/га, RMSE: 27.5692 ц/га
    CNN-LSTM        MAE: 8.4640 ц/га, RMSE: 9.6616 ц/га
    Transformer     MAE: 21.4480 ц/га, RMSE: 22.0274 ц/га
    
    ============================================================
    ОСОБЕННОСТИ ДАННЫХ:
    ============================================================
    • Регионы: Krasnodar Krai, Rostov Oblast, Stavropol Krai, Saratov Oblast, Altai Krai
    • Период: 2010-2023
    • Признаки: NDVI, EVI, температура, осадки, солнечная радиация
    • Целевая переменная: урожайность зерновых (ц/га)
    
    ============================================================
    РЕКОМЕНДАЦИИ:
    ============================================================
    • Для улучшения точности можно добавить больше метеорологических признаков
    • Использовать более длинные временные последовательности
    • Применить ансамблирование моделей
    • Добавить данные о типах почв и сельскохозяйственных практиках
    


```python

```
