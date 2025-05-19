# src/evaluation.py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, log_loss
import matplotlib.pyplot as plt
from config import PLOTS_DIR

def evaluate_model(model, X_test, y_test):
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:', cm)

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc:.4f}'); plt.plot([0,1],[0,1],'--'); plt.savefig(PLOTS_DIR/'roc.png'); plt.close()

    # PR & AP
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(); plt.plot(recall, precision, label=f'AP={ap:.4f}'); plt.savefig(PLOTS_DIR/'pr.png'); plt.close()

    # Log loss
    ll = log_loss(y_test, y_proba)
    print(f'Log Loss: {ll:.4f}')

    return {'cm': cm, 'auc': auc, 'ap': ap, 'log_loss': ll}
