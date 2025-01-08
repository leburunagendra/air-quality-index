# Air Quality Index (AQI) Prediction

This project aims to predict the Air Quality Index (AQI) using various regression models, including Linear Regression, Decision Tree Regressor, Polynomial Regression, and Support Vector Regression (SVR). The dataset used consists of air pollutant components as features and AQI values as the target.

## Prerequisites

Ensure the following Python libraries are installed:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Project Structure

1. **Data Loading**
   - The dataset is loaded from a CSV file provided by the user.
   - Only numeric columns are used as features (excluding the 'AQI' column).
   - Missing values in both features and target are filled with the mean of each column.

2. **Data Splitting**
   - The data is split into training and testing sets (70% training, 30% testing).

3. **Model Evaluation**
   - A function `evaluate_model` is defined to fit a model, make predictions, and compute evaluation metrics:
     - Root Mean Square Error (RMSE)
     - Mean Absolute Error (MAE)
     - Mean Squared Log Error (MSLE)
     - R-squared (R2)

4. **Models Used**
   - Linear Regression
   - Decision Tree Regressor
   - Polynomial Regression (degree 2)
   - Support Vector Regression (SVR) with a linear kernel

5. **Polynomial Features**
   - For Polynomial Regression, the features are transformed to include polynomial terms of degree 2.

6. **Results Storage and Display**
   - The results for each model are stored in a DataFrame and displayed.
   - Predictions for each model are plotted for visual comparison against actual AQI values.

## How to Run

1. Clone or download the project files.
2. Open the script in a Jupyter Notebook or any Python IDE.
3. Run the script and provide the path to your dataset when prompted.
4. Review the printed results and plots for each model's performance.

## Results Interpretation

The script outputs a table with the evaluation metrics for each model. It also generates scatter plots comparing actual and predicted AQI values for each model. Use these outputs to determine the best-performing model based on the metrics and visual accuracy.

## Example Output

| Model                  | RMSE  | MAE   | MSLE  | R2    |
|------------------------|-------|-------|-------|-------|
| Linear Regression      | 10.25 | 7.50  | 0.01  | 0.85  |
| Decision Tree          | 8.15  | 6.20  | 0.008 | 0.90  |
| Polynomial Regression  | 9.75  | 7.30  | 0.009 | 0.87  |
| Support Vector Regression | 11.10 | 8.00 | 0.012 | 0.82  |

## Plots

For each model, a plot is generated showing actual AQI values versus predicted AQI values, helping to visually assess the model's accuracy.


## Acknowledgments

- Thanks to the open-source community for providing valuable tools and libraries that made this project possible.

