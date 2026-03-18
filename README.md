# Travel Insurance Prediction using Neural Networks

## Project Overview

This project builds a **binary classification model** to predict whether a customer will purchase travel insurance based on demographic and behavioral data.

The solution uses a **Neural Network (TensorFlow/Keras)** and follows a structured machine learning pipeline, including:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Model training with regularization
* Evaluation using multiple metrics


## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras


## Dataset

* File: `TravelInsurancePrediction.csv`
* The dataset should be placed in the **same directory** as the Python script.


## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 2. Run the project

```bash
python your_script_name.py
```

---

## Project Workflow

### 1. Data Loading

* Reads dataset using Pandas

### 2. Exploratory Data Analysis (EDA)

* Displays:

  * Head of dataset
  * Data types and structure
  * Statistical summary
  * Missing values
* Visualizes distributions of:

  * Age
  * Annual Income
  * Family Members
  * Target variable (Travel Insurance)


### 3. Data Preprocessing

* Missing values handled using:

  * `fillna(0)`
* Categorical variables encoded using:

  * `pd.factorize()`
* Target variable converted to:

  * One-hot encoded format


### 4. Train-Test Split

* Uses:

  * `train_test_split`
  * Stratification to preserve class distribution


### 5. Feature Scaling (Important)

* Standardization is applied **after splitting**
* Prevents **data leakage**
* Only numerical features are scaled using:

  * `StandardScaler`


### 6. Model Architecture

Neural Network:

* Input layer
* Dense (32 units, ReLU)
* Dropout (0.2)
* Dense (16 units, ReLU)
* Dense (16 units, ReLU)
* Output layer (Softmax)


### 7. Training

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Includes:

  * **EarlyStopping**

    * Monitors validation loss
    * Prevents overfitting
    * Restores best weights


### 8. Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* ROC AUC Score
* ROC Curve


### 9. Training Visualization

* Accuracy vs Epochs
* Loss vs Epochs


### 10. Feature Importance (Approximation)

* Computed using:

  ```
  X.T @ predicted_probabilities
  ```
* Important:

  * This is **NOT true feature importance**
  * Neural networks are **not inherently interpretable**
  * Results should be treated as a **rough approximation only**


## Reproducibility

* Random seeds are set for:

  * NumPy
  * Python random
  * TensorFlow

This ensures consistent results across runs.


## Key ML Considerations

### Data Leakage Prevention

* Scaling is applied **only on training data**
* Validation data is transformed using the same scaler

### Overfitting Control

* EarlyStopping is used during training

### Class Balance

* Stratified splitting ensures fair class distribution


## Example Outputs

* Model performance metrics printed in console
* Graphs:

  * Distributions
  * ROC Curve
  * Training curves
  * Confusion matrix
  * Feature importance (approx.)


## Possible Improvements

* Add baseline models (Logistic Regression, Random Forest)
* Hyperparameter tuning
* Cross-validation
* Use SHAP or LIME for real interpretability
* Deploy model (Flask / FastAPI)


## Example results

<img width="639" height="550" alt="image" src="https://github.com/user-attachments/assets/573226a5-76bd-4189-b1b7-1f4a97141a91" />


<img width="742" height="480" alt="image" src="https://github.com/user-attachments/assets/7a551f9a-f3af-4e48-8362-90fc23fd7812" />


