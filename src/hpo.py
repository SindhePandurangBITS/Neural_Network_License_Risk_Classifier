# hpo.py
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model(trial, input_dim):
    units1 = trial.suggest_int('units1', 32, 256, step=32)
    units2 = trial.suggest_int('units2', 16, 128, step=16)
    dropout1 = trial.suggest_float('dropout1', 0.1, 0.7, step=0.1)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.7, step=0.1)
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    model = Sequential([
        Dense(units1, activation='relu', kernel_initializer=GlorotUniform(), input_dim=input_dim),
        Dropout(dropout1),
        Dense(units2, activation='relu', kernel_initializer=GlorotUniform()),
        Dropout(dropout2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[AUC()])
    return model

def objective(trial, X_train, y_train, X_val, y_val, input_dim):
    model = create_model(trial, input_dim)
    es = EarlyStopping(monitor='val_auc', mode='max', patience=2, restore_best_weights=True)
    pruning_cb = optuna.integration.TFKerasPruningCallback(trial, 'val_auc')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=10, batch_size=64, callbacks=[es, pruning_cb], verbose=0)
    score = model.evaluate(X_val, y_val, verbose=0)[1]
    return score

def run_study(X_train, y_train, X_val, y_val, input_dim):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, input_dim), n_trials=20, timeout=1800)
    print('Best AUC:', study.best_value)
    print('Best params:', study.best_params)
    return study.best_params
