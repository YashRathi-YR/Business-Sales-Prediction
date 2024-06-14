import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

class MLModelApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ML Model with SARIMA')
        self.setGeometry(100, 100, 1200, 600)

        layout = QVBoxLayout()

        self.file_label = QLabel('No file selected.')
        layout.addWidget(self.file_label)

        file_button = QPushButton('Select CSV File')
        file_button.clicked.connect(self.load_csv)
        layout.addWidget(file_button)

        self.sales_column_label = QLabel('Select Sales Column:')
        layout.addWidget(self.sales_column_label)

        self.sales_column_dropdown = QComboBox()
        layout.addWidget(self.sales_column_dropdown)

        self.time_column_label = QLabel('Select Time Column:')
        layout.addWidget(self.time_column_label)

        self.time_column_dropdown = QComboBox()
        layout.addWidget(self.time_column_dropdown)

        self.seasonality_label = QLabel('Select Seasonality:')
        layout.addWidget(self.seasonality_label)

        self.seasonality_dropdown = QComboBox()
        self.seasonality_dropdown.addItems(['Monthly', 'Quarterly', 'Daily'])
        layout.addWidget(self.seasonality_dropdown)

        train_button = QPushButton('Train SARIMA Model')
        train_button.clicked.connect(self.train_sarima)
        layout.addWidget(train_button)

        self.setLayout(layout)
        self.show()

    def load_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select CSV File', '', 'CSV Files (*.csv)')

        if file_path:
            self.file_label.setText(f'Selected File: {file_path}')
            self.df = pd.read_csv(file_path)

            self.sales_column_dropdown.addItems(self.df.columns)
            self.time_column_dropdown.addItems(self.df.columns)

    def train_sarima(self):
        sales_column = self.sales_column_dropdown.currentText()
        time_column = self.time_column_dropdown.currentText()

        # Step 6: Data Cleaning
        self.df[time_column] = pd.to_datetime(self.df[time_column], errors='coerce')
        self.df = self.df.dropna(subset=[time_column])  # Drop rows with non-parsable time data
        self.df = self.df.set_index(time_column).sort_index()

        # Handle missing values and empty columns
        self.df = self.df.ffill()

        # Ensure datetime index has a frequency
        self.df = self.df.asfreq('MS')  # Monthly start frequency, adjust if needed

        # Check if we have enough data points
        if len(self.df) < 20:  # Example threshold
            print("Not enough data points to train the model.")
            return

        # Step 7: Split the data into train and test sets
        train_size = int(len(self.df) * 0.8)  # 80% for training, 20% for testing
        train, test = self.df.iloc[:train_size], self.df.iloc[train_size:]

        # Step 8: SARIMA Model Training
        order = (1, 1, 1)  # You may adjust the order based on your data and analysis
        seasonal_order = self.get_seasonal_order()

        model = SARIMAX(train[sales_column], order=order, seasonal_order=seasonal_order)
        results = model.fit()

        # Step 9: Forecasting
        forecast_steps = len(test)  # Forecast the same number of steps as in the test set
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_index = test.index
        forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

        # Step 10: Display Insights
        self.plot_autocorrelation(results.resid)
        self.plot_forecast(train[sales_column], test[sales_column], forecast_series)

    def get_seasonal_order(self):
        seasonality = self.seasonality_dropdown.currentText()

        if seasonality == 'Monthly':
            return (1, 1, 1, 12)
        elif seasonality == 'Quarterly':
            return (1, 1, 1, 4)
        elif seasonality == 'Daily':
            return (1, 1, 1, 365)

    def plot_autocorrelation(self, residuals):
        plot_acf(residuals, lags=20, title='Autocorrelation Plot of Residuals')
        plt.show()

    def plot_forecast(self, train_series, test_series, forecast_series):
        plt.figure(figsize=(12, 6))
        plt.plot(train_series.index, train_series, label='Training Data')
        plt.plot(test_series.index, test_series, label='Test Data')
        plt.plot(forecast_series.index, forecast_series.values, label='Forecasted Sales', linestyle='dashed')
        plt.legend()
        plt.title('SARIMA Model Forecast')
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ml_app = MLModelApp()
    sys.exit(app.exec_())
