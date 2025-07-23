# UCC Hostel Rent Predictor

This repository contains the capstone project for an MSc in Data Management and Analysis at the University of Cape Coast (UCC). The project develops a machine learning model to predict hostel rent prices for UCC students, deployed as an interactive Streamlit web application. The goal is to provide data-driven insights to help students make informed accommodation decisions.

## Project Overview

The **UCC Hostel Rent Predictor** uses a machine learning model to estimate annual hostel rent prices based on features such as room type, amenities, location, and deposit. The best-performing model, XGBoost, achieves an R² of 0.9769 and RMSE of 198.56 GHS, ensuring accurate predictions. The model is integrated into a Streamlit web app, allowing users to input hostel details and receive real-time rent predictions.

### Objectives

- Develop a robust machine learning model for hostel rent prediction.
- Identify key factors influencing rent prices through exploratory data analysis.
- Deploy the model as a user-friendly web application for students and stakeholders.
- Contribute to transparent and affordable housing solutions at UCC.

### Key Features

- **Dataset**: 502 records with 30 features (e.g., deposit, room size, wifi, hostel location).
- **Model**: XGBoost with hyperparameter tuning (e.g., `n_estimators=300`, `learning_rate=0.2`).
- **Web App**: Streamlit interface for interactive rent predictions.
- **Insights**: Deposit, average area rent, and room size are the top predictors of rent.

## Repository Structure

```
UCC-Hostel-Rent-Predictor/
├── .devcontainer/              # Configuration for development environment
├── Data/                       # Dataset files
│   └── Hostel_Prices.csv       # Raw dataset
├── model/                      # Trained model and sample input
│   ├── ucc_hostel_rent_predictor.pkl
│   └── hostel_sample_input.pkl
├── modelling/                  # Jupyter notebooks and scripts
│   └── notebook.ipynb          # Main analysis and model training
├── rent_app.py                 # Streamlit web application
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Git
- Streamlit
- A GitHub account for cloning the repository

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/UCC-Hostel-Rent-Predictor.git
   cd UCC-Hostel-Rent-Predictor
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:

   ```
   pandas
   numpy
   scikit-learn
   xgboost
   streamlit
   joblib
   matplotlib
   seaborn
   ```

4. **Run the Streamlit App**:

   ```bash
   streamlit run rent_app.py
   ```

   This will launch the web app in your default browser at `http://localhost:8501`.

## Usage

### Running the Analysis

- Open `modelling/notebook.ipynb` in Jupyter Notebook to explore the data analysis, preprocessing, and model training.
- The notebook includes:
  - Data cleaning and feature engineering.
  - Exploratory data analysis (e.g., histograms, correlation heatmaps).
  - Model training and evaluation (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost).

### Using the Streamlit App

1. Navigate to the "Predict Hostel Rent" page via the sidebar.
2. Enter hostel details:
   - **Numerical Features**: Deposit, room size, commute time, etc.
   - **Categorical Features**: Gender, study level, room type, etc.
   - **Amenities**: Checkboxes for wifi, security, water, etc.
3. Click "Predict Rent" to view the predicted annual rent in GHS.

### Example Input

- **Deposit**: 2500 GHS
- **Room Type**: Private room
- **Hostel Location**: Kwaprow
- **Amenities**: Wifi, security, water included
- **Output**: Predicted Annual Rent: \~GHS 4000 (example)

## Methodology

### Data Preprocessing

- **Cleaning**: Standardized column names, corrected inconsistencies (e.g., "Post Graduate" to "Postgraduate").
- **Imputation**: Group-based median for numerical features, mode for categorical features.
- **Feature Engineering**:
  - Binary encoding for amenities.
  - Created furnishing score, total amenities, rent-to-deposit ratio, and distance-weighted rent.
  - Log-transformed skewed features (e.g., annual rent, deposit).
- **Pipeline**: Used `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.

### Model Development

- **Models Tested**: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost.
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation for tree-based models.
- **Best Model**: XGBoost (`n_estimators=300`, `learning_rate=0.2`, `max_depth=3`).

### Evaluation

- **Metrics**:
  - RMSE: 198.56 GHS (XGBoost)
  - R²: 0.9769 (XGBoost)
  - Cross-Validation RMSE: 0.10
- **Feature Importance**: Deposit, average area rent, and room size were the top predictors.

### Deployment

- The XGBoost model is saved as `ucc_hostel_rent_predictor.pkl`.
- A Streamlit app (`rent_app.py`) provides an interactive interface for real-time predictions.

## Results

- **Model Performance**:
  - XGBoost: RMSE = 198.56 GHS, R² = 0.9769
  - Gradient Boosting: RMSE = 240.47, R² = 0.9662
  - Random Forest: RMSE = 339.67, R² = 0.9325
- **Key Insights**:
  - Deposit strongly correlates with rent (0.912).
  - Local market conditions (average area rent) and room size are critical drivers.
- **Web App**: Enables students to predict rent with an intuitive interface.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Authors

- **Mohammed Kamalidin**
  - Email: mkamalidin9@gmail.com
  - LinkedIn: Mohammed Kamalidin
  - Affiliation: University of Cape Coast, MSc Data Management and Analysis
- **Prince Acquah Rockson**
  - Email: parockson@gmail.com
  - LinkedIn: Prince Acquah Rockson
  - Affiliation: Kwame Nkrumah University of Science and Technology, MSc Health Informatics

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Data contributors for providing the `Hostel_Prices.csv` dataset
