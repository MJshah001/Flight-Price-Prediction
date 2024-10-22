# Flight Price Prediction [ AWS Sagemaker, Flask API, Streamlit App & Docker ]

## Project Overview
This project focuses on predicting flight prices using machine learning techniques. The dataset used for this project includes various features such as airline, source, destination, departure time, and more. The model was trained using XGBoost on AWS SageMaker and deployed as a REST API using Flask, a web application using Streamlit, and a Docker container for advanced deployment.

##  Live Demo
Flask APP and API : [click here](https://flask-ml-project-flight-price-prediction.onrender.com/)

Streamlit APP : [click here](https://flight-price-prediction-aws-sagemaker-machine-learning-project.streamlit.app/)

Docker Image : [click here](https://hub.docker.com/r/mjshah001/flight-price-prediction-app)

## Project Architecture

![project architecture](https://github.com/MJshah001/Flight-Price-Prediction/blob/master/screenshots/flight%20price%20prediction%20project%20architecture.jpg)

The project is divided into three major sections: **Data Preprocessing & Feature Engineering**, **Model Training**, and **Model Serving & Deployment**.


### 1. Data Preprocessing & Feature Engineering
- **Environment:** Local Machine
- **Files:** 
  - `data_cleaning.ipynb`: Handles missing values, duplicates, and basic data cleaning.
  - `EDA.ipynb`: Explores the data to understand the relationships between features and the target variable.
  - `feature_engineering.ipynb`: Constructs new features and transforms existing ones to improve model performance.
  - `eda_helper_functions.py`: Contains helper functions used in the EDA notebook.
  - **Data Folder:**
    - `train.csv`, `test.csv`, `validation.csv`: The split datasets used for training, validation, and testing.

### 2. Model Training
- **Environment:** AWS cloud
- **Files:**
  - `model_training.ipynb`: Trains the XGBoost model using the preprocessed data.
  - `train_preprocessed`, `test_preprocessed`, `validation_preprocessed`: Processed datasets used for training and evaluating the model.
  - `preprocessor.pkl`: The serialized preprocessor used for transforming new data.
  - `xgboost-model`: The final trained model.

### 3. Model Serving & Deployment
- **Environment:** Local Machine or WEB
- **Files:**
  - `app.py`: Contains the Flask API to serve predictions.
  - `streamlit-app.py`: Contains the Streamlit App to serve predictions.
  - `custom_functions.py`: Custom preprocessing functions used in both Flask and Streamlit app.
  - `forms.py`: Manages the form inputs for the web interface.
  - `requirements.txt`: Lists all dependencies required to run the Flask app.
  - `streamlit-requirements.txt`: Lists all dependencies required to run the Streamlit app.
  - **Templates Folder:**
    - `home.html`, `layout.html`, `predict.html`, `api_guide.html`: HTML files for the Flask web interface.


## Project Setup
### Prerequisites
- Python 3.8 or later
- Anaconda or any Python environment manager
- Jupyter Notebook (for Data Wrangling and EDA)
- AWS account with access to SageMaker (for Model Training)
- Libraries mentioned in `requirements.txt`/`streamlit-requirements.txt`
- Free Streamlit Cloud Account : https://share.streamlit.io/ (for Cloud Deployment)
- Free Render Account : https://dashboard.render.com/ (for Cloud Deployment)

Note: refer to [Screenshots Folder](https://github.com/MJshah001/Flight-Price-Prediction/tree/master/screenshots) for more information regaring setup.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MJshah001/Flight-Price-Prediction.git
   cd flight-price-prediction
2. Create a Virtual Environment
   ```bash
   conda create --name flight-price-env python=3.8
   conda activate flight-price-env
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -r streamlit-requirements.txt

### Data Preprocessing & Feature Engineering
1. Run the `data_cleaning.ipynb` notebook to clean the data.
   - SET `PROJECT_DIR` : Define the directory path where your project is located.
   - SET `DATA_DIR` : Specify the name of the folder containing the CSV data files. 
2. Perform exploratory data analysis using `EDA.ipynb`.
   - SET `file_path` : Specify the full path to the training data file.
3. Generate and save new features with `feature_engineering.ipynb`.
   - SET `file_path` : Specify the full path to the training data file.


### Model Training
1. Set up the AWS sagemaker Notebook instance and upload model_training.ipynb, train, test & validation csv files to instance.
2. create your AWS S3 bucket to store preprocess data and trained model.
3. Open `model_training.ipynb` on SageMaker and train the XGBoost model.
   - SET `BUCKET_NAME` : Specify the name of your newly created S3 bucket.
4. Execute the `model_training.ipynb` to save the trained model and preprocessor to the S3 bucket for deployment.
5. Download trained `xgboost-model` and `preprocessor.pkl` for deployment.

Note: refer to [Screenshots Folder](https://github.com/MJshah001/Flight-Price-Prediction/tree/master/screenshots) for more information regaring Model training steps.

### Model Serving & Deployment
1. setup `app.py` file to create flask app.
2. To deploy locally using Flask, run:
   ```bash
   python app.py
3. setup `streamlit-app.py` file to create streamlit app.
4. To deploy locally using Streamlit, run:
   ```bash
   streamlit run streamlit-app.py
5. To deploy flask app on cloud
    - Login to Render Dashboard : https://dashboard.render.com/
    - Create a New Web service
        - provide unique `Name` for your webservice.
        - Select Free Instance Type `Free 0.1 CPU 512 MB`.
        - Provide your repositry URL or my public url `https://github.com/MJshah001/Flight-Price-Prediction`.
        - Provide build command as `pip install -r requirements.txt`.
        - Provide Start command as `gunicorn app:app`.
        - Click `Deploy Web Service`.
6. To deploy streamlit app on cloud
    - Login to Streamlit share : https://share.streamlit.io/
    - click on create app.
       - Provide your repositry URL or my public url `https://github.com/MJshah001/Flight-Price-Prediction`.
       - Provide `Main file path` as `streamlit-app.py`.
       - click `Deploy`.
7. Docker Deployment
      1. **Build the Docker image locally:**
    ```bash
    docker build -t flight-price-prediction-app .
    ```
      2. **Push the Docker image to Docker Hub:**
    ```bash
    docker push mjshah001/flight-price-prediction-app
    ```
      3. **Pull and run the Docker image:**
    ```bash
    docker pull mjshah001/flight-price-prediction-app
    docker run -p 5000:5000 mjshah001/flight-price-prediction-app
    ```

    
## Installation via Docker

To simplify the installation process and run the Flight Price Prediction application using Docker, follow these steps:

1. **Ensure Docker is Installed:**
   - If Docker is not installed, download and install it from the [official Docker website](https://www.docker.com/get-started).

2. **Pull the Docker Image:**
   - To obtain the pre-built Docker image for the Flight Price Prediction application, use the following command:
     ```bash
     docker pull mjshah001/flight-price-prediction-app
     ```

3. **Run the Docker Container:**
   - Start a new container from the pulled Docker image and map the container's port to your local machine's port. This allows you to access the application from your browser:
     ```bash
     docker run -p 5000:5000 mjshah001/flight-price-prediction-app
     ```
   - Here, `5000:5000` maps port 5000 on your local machine to port 5000 on the Docker container. Adjust the ports if necessary.

4. **Access the Application:**
   - Open your web browser and navigate to `http://localhost:5000` to access the Flask API and web interface.
   - For the Streamlit app, if you have deployed it separately, use the provided Streamlit app URL.

5. **Stop the Docker Container:**
   - To stop the running container, find the container ID using:
     ```bash
     docker ps
     ```
   - Then stop the container with:
     ```bash
     docker stop <container_id>
     ```
   - Replace `<container_id>` with the actual ID of the running container.

By using Docker, you can easily run the Flight Price Prediction application in a consistent environment without worrying about dependencies and configurations.


## Flight Price Prediction API

This section will help you understand how to interact with the API endpoint for predicting flight prices.

![postman api test](https://github.com/MJshah001/Flight-Price-Prediction/blob/master/screenshots/Postman%20API%20test.png)

### API Endpoint

```bash
https://flask-ml-project-flight-price-prediction.onrender.com/api/predict
```

### Request Format
The request should have the following format and include the Content-Type: `application/json` in headers:

```bash
{
    "airline": "Air India",
    "date_of_journey": "2024-08-16",
    "source": "Delhi",
    "destination": "Cochin",
    "dep_time": "13:20:00",
    "arrival_time": "15:20:00",
    "duration": 2000,
    "total_stops": 10,
    "additional_info": "Business Class"
}
```
### Inputs

Here are the possible values and formats for each input field:

- **Airline**:
    - Air India
    - Jet Airways
    - Indigo
    - Multiple Carriers
    - Spicejet
    - Vistara
    - Air Asia
    - Goair
    - Trujet

- **Date of Journey**:
    - Format: `YYYY-MM-DD`

- **Source**:
    - Delhi
    - Kolkata
    - Banglore
    - Mumbai
    - Chennai

- **Destination**:
    - Cochin
    - Banglore
    - New Delhi
    - Delhi
    - Hyderabad
    - Kolkata

- **Departure Time**:
    - Format: `HH:MM:SS`

- **Arrival Time**:
    - Format: `HH:MM:SS`

- **Duration**:
    - Format: `Integer (in minutes)`

- **Total Stops**:
    - Format: `Integer`

- **Additional Info**:
    - No Info
    - In-flight meal not included
    - 1 Long layover
    - No check-in baggage included
    - Change airports
    - Red-eye flight
    - Business class
    - 1 Short layover
    - 2 Long layover


### Example Request Using curl :

`curl` is a command-line tool used for making HTTP requests. Here’s an example of how to use `curl` to send a request to the API:

```bash
curl -X POST https://flask-ml-project-flight-price-prediction.onrender.com/api/predict \
     -H "Content-Type: application/json" \
     -d '{
         "airline": "Air India",
         "date_of_journey": "2024-08-16",
         "source": "Delhi",
         "destination": "Cochin",
         "dep_time": "13:20:00",
         "arrival_time": "15:20:00",
         "duration": 2000,
         "total_stops": 10,
         "additional_info": "Business Class"
     }'

```

### Response Format

The API will return a JSON response with the predicted flight price. An example response might look like this:

```bash
{
    "prediction": 13435.7
}
```
Find more details on https://flask-ml-project-flight-price-prediction.onrender.com/api/help


## Flight Price Prediction using UI

Option 1. Flask app UI: https://flask-ml-project-flight-price-prediction.onrender.com/predict

![Flask predict UI](https://github.com/MJshah001/Flight-Price-Prediction/blob/master/screenshots/flask%20predict%20UI.png)

Option 2: Streamlit app: https://flight-price-prediction-aws-sagemaker-machine-learning-project.streamlit.app/

![Streamlit predict UI](https://github.com/MJshah001/Flight-Price-Prediction/blob/master/screenshots/streamlit%20app.png)


## Future Work

- Improve model accuracy with additional features and data.
- Integrate with more advanced deployment options like Docker or Kubernetes.
- Develop an API endpoint & user interface on the web app for model retraining with new training data.

## Conclusion

This project demonstrated the entire lifecycle of a data science machine learning project, from data preprocessing and model training to deployment and API integration. It served as a valuable learning experience in handling real-world data and deploying machine learning models.

## Author

- **Monil Shah** - _Master's Student in Data Science, NJIT_



