PROJECT TITLE
- Red Wine Quality Prediction using ANN

PROJECT OVERVIEW
- This project applies Deep Learning techniques to predict the quality of red wine based on physicochemical tests. Using an Artificial Neural Network (ANN) built with Keras and TensorFlow, the model analyzes 11 chemical properties (like acidity, sugar, pH, and alcohol) to output a continuous quality score. The goal is to assist winemakers in assessing quality consistency using data-driven methods.

DATASET DESCRIPTION
The dataset ("winequality-red.csv") contains 1,599 samples of red wine variants with the following input features:
1. Fixed Acidity: Non-volatile acids involved with wine.
2. Volatile Acidity: Amount of acetic acid (too high leads to vinegar taste).
3. Citric Acid: Adds freshness and flavor.
4. Residual Sugar: Amount of sugar remaining after fermentation stops.
5. Chlorides: Amount of salt in the wine.
6. Free Sulfur Dioxide: Prevents microbial growth and oxidation.
7. Total Sulfur Dioxide: Amount of free + bound forms of SO2.
8. Density: Depends on percent alcohol and sugar content.
9. pH: Describes how acidic or basic a wine is (scale 0-14).
10. Sulphates: Additive which can contribute to sulfur dioxide gas (S02) levels.
11. Alcohol: The percent alcohol content of the wine.
12. Quality (Target): A score between 3 and 8 based on sensory data.

OBJECTIVES OF THE PROJECT
- To perform Exploratory Data Analysis (EDA) to understand feature distributions and correlations.
- To detect and handle outliers using the IQR (Interquartile Range) method.
- To preprocess the data by splitting features and targets, and applying scaling techniques.
- To build a regression-based Artificial Neural Network (ANN) to predict the quality score.
- To evaluate the model's performance using Mean Squared Error (MSE).

STEPS PERFORMED
1. Data Loading & Inspection: Loaded the CSV file using Pandas, checked for null values, and generated descriptive statistics.
2. Exploratory Data Analysis (EDA):
   - Visualized the distribution of wine quality scores using count plots.
   - Created correlation heatmaps to identify relationships between chemical features and quality.
   - Used boxplots to detect outliers and visualized feature distributions.
   - Plotted relationships like Alcohol vs. Density and Alcohol vs. Quality.
3. Data Cleaning:
   - Applied the IQR method to remove significant outliers, ensuring a cleaner dataset for training.
4. Model Architecture:
   - Constructed a Sequential ANN model using Keras.
   - Added multiple Dense layers with 'relu' activation functions.
   - Used a final linear activation layer for regression output (predicting a continuous score).
5. Model Training & Evaluation:
   - Split data into training (70%) and testing (30%) sets.
   - Compiled the model using the 'adam' optimizer and 'mean_squared_error' loss.
   - Trained for 50 epochs and evaluated performance on the test set.

TOOLS AND LIBRARIES USED
- Python
- Pandas (for data manipulation)
- NumPy (for numerical computations)
- Matplotlib & Seaborn (for data visualization)
- Scikit-learn (for train-test splitting)
- TensorFlow & Keras (for building and training the neural network)
- Jupyter Notebook

FILES INCLUDED
- wine_quality.ipynb: The Jupyter Notebook containing the full analysis and model code.
- winequality-red.csv: The dataset used for training and testing.

HOW TO RUN THE PROJECT
1. Install the required libraries:
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
2. Download "wine_quality.ipynb" and "winequality-red.csv" to the same directory.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Run the cells sequentially to visualize data, train the ANN, and see prediction results.

KEY INSIGHTS
- Feature Correlation: Alcohol content has the highest positive correlation with wine quality, meaning higher alcohol content generally relates to better quality ratings.
- Volatile Acidity: This feature has a strong negative correlation with quality; higher acidity levels tend to lower the wine's quality score.
- Model Performance: The ANN successfully learned from the chemical properties to predict quality scores, with performance verified by the Test Mean Squared Error (MSE).
- Data Distribution: The dataset is slightly imbalanced, with most wine samples having 'average' quality scores (5 or 6).

CONCLUSION
- This project demonstrates that physicochemical properties can effectively predict wine quality. By leveraging Deep Learning, we can automate the quality grading process, providing a scientific backup to human sensory testing. The identified key features (Alcohol, Volatile Acidity) offer critical insights for production control.

AUTHOR
- Rishan Menezes
