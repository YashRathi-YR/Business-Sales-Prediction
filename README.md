Business Sales Forecasting Web App with SARIMA Model

Welcome to my Business Prediction project repository! In this project, I've developed a website using Flask API and implemented a SARIMA time series forecasting model. I utilized advanced data preprocessing techniques and feature selection methods in Python to enhance model accuracy. 

Demonstrating my analytical skills, I evaluated the model's performance using metrics such as Mean Absolute Error. The deployed models provide practical insights and support decision-making processes for business applications.

This project is designed to work with various types of CSV files, making it versatile and suitable for a wide range of business datasets.

Feel free to explore the code and project files. If you have any questions or suggestions, please don't hesitate to reach out. Thank you for visiting!

###To run the provided Flask application, follow these steps:

 Setting Up the Environment:
1. Install Python: Make sure you have Python installed on your system. You can download it from the official Python website: [python.org](https://www.python.org/downloads/).

2. Install Flask and Dependencies: Use pip, the Python package installer, to install Flask and other required libraries. Run the following command in your terminal or command prompt:
    ```
    pip install flask statsmodels pandas matplotlib scikit-learn
    ```

 Running the Application:
3. Organize Your Project: Ensure that your project directory structure looks like this:
    ```
    project_folder/
    │
    ├── app.py
    ├── static/
    │   └── (static files like CSS, JavaScript)
    ├── templates/
    │   └── index.html
    └── uploads/
        └── (uploaded CSV files)
    ```

4. Navigate to Project Directory: Open a terminal or command prompt and navigate to the directory where your `app.py` file is located.

5. Run the Application: Run the Flask application by executing the following command:
    ```
    python app.py
    ```

6. Access the Application: Once the Flask app is running, open a web browser and go to `http://localhost:5000` to access the application.

 Using the Application:
7. Upload CSV File: Click on the "Choose File" button to select a CSV file containing your sales data.

8. Select Columns and Seasonality: Choose the column containing sales data and the column containing time information from the dropdown menus. Also, select the seasonality (Monthly, Quarterly, or Daily).

9. Click on Predict: After selecting the necessary options, click on the "Predict" button.

10. View Results: The application will display the forecasted sales, autocorrelation plot, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and comparison plot of actual vs. predicted values.

 Additional Notes:
- Ensure the CSV file is properly formatted and contains the selected columns.
- If any errors occur during the process, appropriate error messages will be displayed on the webpage.
- You can modify the HTML template (`index.html`) or the Flask application (`app.py`) to customize the appearance or functionality according to your requirements.
