from flask import Flask, render_template, request
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Configure logging (you can adjust the level and format)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure 'uploads' folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class MLModel:
    def __init__(self):
        self.df = pd.DataFrame()
        self.results = None
        self.forecast_series = None

ml_model = MLModel()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            error_message = "No file selected. Please choose a valid CSV file."
            return render_template('index.html', error_message=str(error_message))

        # Save the uploaded CSV file to the 'uploads' folder
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')
        uploaded_file.save(csv_path)

        ml_model.df = pd.read_csv(csv_path)
        ml_model.sales_column = request.form['sales_column']
        ml_model.time_column = request.form['time_column']
        ml_model.seasonality = request.form['seasonality']
        ml_model.results, ml_model.forecast_series = train_sarima()

        ml_model.forecast_series.index = pd.to_datetime(ml_model.forecast_series.index)

        # Get autocorrelation plot as a base64-encoded image
        acf_img_base64 = base64.b64encode(get_autocorrelation_plot().read()).decode('utf-8')

        # Calculate statistical metrics
        actual_values = ml_model.df[ml_model.sales_column][-12:]
        mae = mean_absolute_error(actual_values, ml_model.forecast_series)
        rmse = mean_squared_error(actual_values, ml_model.forecast_series, squared=False)

        # Generate a plot comparing actual vs. predicted values
        comparison_plot = plot_actual_vs_predicted(actual_values, ml_model.forecast_series)

        # Encode the forecast plot and comparison plot to base64 for embedding in HTML
        forecast_plot_base64 = base64.b64encode(get_forecast_plot(ml_model.forecast_series).read()).decode('utf-8')
        comparison_plot_base64 = base64.b64encode(comparison_plot.read()).decode('utf-8')

        return render_template('index.html', forecast_series=ml_model.forecast_series, acf_img=acf_img_base64,
                                mae=mae, rmse=rmse, comparison_plot=comparison_plot_base64,
                                forecast_plot=forecast_plot_base64)

    except pd.errors.EmptyDataError:
        error_message = "The uploaded CSV file is empty. Please choose a valid file."
    except (KeyError, ValueError) as e:
        error_message = f"An error occurred: {e}. Please check your input data."
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Please try again or contact support."
        logging.error(f"An error occurred: {e}")

    return render_template('index.html', error_message=str(error_message))
def train_sarima():
    try:
        sales_column = ml_model.sales_column
        time_column = ml_model.time_column

        if not auto_parse_date(time_column):
            logging.error("Error: Unable to parse dates. Check the date formats.")
            return None, None

        ml_model.df = ml_model.df.dropna(subset=[time_column])
        ml_model.df[time_column] = pd.to_datetime(ml_model.df[time_column])
        ml_model.df = ml_model.df.set_index(time_column).sort_index()

        ml_model.df = ml_model.df.ffill()

        order = (1, 1, 1)
        seasonal_order = get_seasonal_order()

        model = SARIMAX(ml_model.df[sales_column], order=order, seasonal_order=seasonal_order)
        results = model.fit(method='powell')  

        forecast_steps = 12
        forecast_index = pd.date_range(start=ml_model.df.index[-1], periods=forecast_steps + 1, freq='M')[1:]
        forecast_series = pd.Series(results.get_forecast(steps=forecast_steps).predicted_mean.values, index=forecast_index)

        ml_model.results = results
        ml_model.forecast_series = forecast_series

        return results, forecast_series
    except Exception as e:
        logging.error(f"An error occurred in train_sarima: {e}")
        return None, None

def auto_parse_date(time_column):
    potential_formats = ['%Y-%m-%d', '%Y%m%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%Y-%m']

    for date_format in potential_formats:
        try:
            parsed_dates = pd.to_datetime(ml_model.df[time_column], format=date_format, errors='coerce')
            if not parsed_dates.isnull().any():
                ml_model.df[time_column] = parsed_dates
                return True
        except ValueError:
            pass

    ml_model.df[time_column] = ml_model.df[time_column].apply(lambda x: custom_date_parser(x, time_column))
    parsed_dates = pd.to_datetime(ml_model.df[time_column], errors='coerce')
    if not parsed_dates.isnull().any():
        ml_model.df[time_column] = parsed_dates
        return True

    logging.error(f"Error: Unable to parse dates in column '{time_column}'. Check the date formats.")
    return False

def custom_date_parser(date_str, time_column):
    try:
        parsed_date = pd.to_datetime(date_str, errors='coerce')
        if not pd.isnull(parsed_date):
            ml_model.df[time_column] = parsed_date
            return parsed_date

        default_year = 2023
        date_with_default_year = f"{default_year}-{date_str}"

        parsed_date_with_default_year = pd.to_datetime(date_with_default_year, errors='coerce')
        if not pd.isnull(parsed_date_with_default_year):
            ml_model.df[time_column] = parsed_date_with_default_year
            return parsed_date_with_default_year

        parsed_date_year_month = pd.to_datetime(date_str + "-01", format="%Y-%m-%d", errors='coerce')
        if not pd.isnull(parsed_date_year_month):
            ml_model.df[time_column] = parsed_date_year_month
            return parsed_date_year_month

    except Exception as e:
        logging.error(f"Error parsing date: {e}")

    return None

def get_seasonal_order():
    seasonality = ml_model.seasonality

    if seasonality == 'Monthly':
        return (1, 1, 1, 12)
    elif seasonality == 'Quarterly':
        return (1, 1, 1, 4)
    elif seasonality == 'Daily':
        return (1, 1, 1, 365)

def get_autocorrelation_plot():
    fig = Figure(figsize=(8, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    plot_acf(ml_model.results.resid, lags=20, ax=ax)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

def get_forecast_plot(forecast_series):
    fig, ax = plt.subplots()
    ax.plot(forecast_series, label='Forecast')
    ax.legend()
    plt.xlabel('Time')
    plt.ylabel('Sales')
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

def plot_actual_vs_predicted(actual_values, predicted_values):
    fig, ax = plt.subplots()
    ax.plot(actual_values, label='Actual')
    ax.plot(predicted_values, label='Predicted')
    ax.legend()
    plt.xlabel('Time')
    plt.ylabel('Sales')
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

if __name__ == '__main__':
    
    app.run(debug=True)
