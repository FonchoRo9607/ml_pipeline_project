from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """Genera métricas de validación."""
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    return report