# architecture.py
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform

def build_model(units1=128, units2=64, dropout1=0.5, dropout2=0.5, input_dim=None):
    model = Sequential([
        Dense(units1, activation='relu', kernel_initializer=GlorotUniform(), input_dim=input_dim),
        Dropout(dropout1),
        Dense(units2, activation='relu', kernel_initializer=GlorotUniform()),
        Dropout(dropout2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'AUC']
    )
    model.summary()
    return model