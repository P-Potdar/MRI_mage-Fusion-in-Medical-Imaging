import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

# Generate synthetic dataset for demonstration
def generate_data(num_samples=500):
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(num_samples) * 100,
        'feature2': np.random.rand(num_samples) * 100,
        'feature3': np.random.rand(num_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C'], num_samples),
    })
    X['target_class'] = (X['feature1'] + X['feature2'] + np.random.randn(num_samples) * 10 > 100).astype(int)
    X['target_reg'] = X['feature1'] * 0.5 + X['feature2'] * 0.3 + np.random.randn(num_samples) * 10
    return X

# Dataset visualization functions
def visualize_dataset(data):
    # Bar Chart and Pie Chart for categorical columns
    for column in data.select_dtypes(include=['object', 'category']).columns:
        plt.figure(figsize=(10, 5))
        data[column].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Counts')
        plt.savefig(f'bar_chart_{column}.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        data[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Pie Chart of {column}')
        plt.ylabel('')
        plt.savefig(f'pie_chart_{column}.png')
        plt.close()

    # Histogram for numerical columns
    for column in data.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'histogram_{column}.png')
        plt.close()

    # Correlation heatmap (only for numerical columns)
    plt.figure(figsize=(12, 8))
    numeric_cols = data.select_dtypes(include=['number']).columns
    sns.heatmap(data[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Distribution of numerical columns
    for column in data.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data[column], fill=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.savefig(f'distribution_{column}.png')
        plt.close()

    # Pairplot for numerical features
    sns.pairplot(data.select_dtypes(include=['number']))
    plt.title('Pairplot of Numerical Features')
    plt.savefig('pairplot_numerical_features.png')
    plt.close()

# Classification model evaluation
def evaluate_classification_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('precision_recall_curve.png')
    plt.close()

    # Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test Score')
    plt.title('Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.close()

    # Feature Importance
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances, y=X_train.columns)
    plt.title('Feature Importance Plot')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig('feature_importance.png')
    plt.close()

# Regression model evaluation
def evaluate_regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')

    # Prediction vs Actual
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Prediction vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('prediction_vs_actual.png')
    plt.close()

    # Residuals Plot
    plt.figure(figsize=(8, 5))
    sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True)
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig('residuals_plot.png')
    plt.close()

    # Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test Score')
    plt.title('Learning Curves for Regression')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('learning_curves_regression.png')
    plt.close()

    # Q-Q Plot of Residuals
    residuals = y_test - y_pred
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.savefig('qq_plot_residuals.png')
    plt.close()

def main():
    # Generate dataset
    data = generate_data()

    # Visualize dataset
    visualize_dataset(data)

    # Prepare data for classification
    X_class = data[['feature1', 'feature2', 'feature3']]
    y_class = data['target_class']
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    # Evaluate classification model
    evaluate_classification_model(X_train_class, X_test_class, y_train_class, y_test_class)

    # Prepare data for regression
    X_reg = data[['feature1', 'feature2']]
    y_reg = data['target_reg']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Evaluate regression model
    evaluate_regression_model(X_train_reg, X_test_reg, y_train_reg, y_test_reg)

if __name__ == "__main__":
    main()
