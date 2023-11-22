## Simple Image Classifier API using FastAPI

This FastAPI application serves as an Image Classifier API, allowing users to make predictions on images.

### Features

- **Predictions:** Provides predictions on input images using a pre-trained image classifier.

### Setup

### Prerequisites

Make sure you have the following installed before running the application:

- Python (>= 3.7)
- Poetry (environment mangement)
- FastAPI and Uvicorn
- Required dependencies (specified in pyproject.toml)


## Installation

1. Install dependencies:

    ```bash
    poetry install --only main
    ```

2. Navigate to "src" directory:

    ```bash
    cd src
    ```

2. Run the FastAPI application:

    ```bash
    poetry run python main.py
    ```

3. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore available endpoints and make predictions.

### API Endpoints

#### 1. Hello World

- **Endpoint:** `/`
- **Method:** `GET`
- **Description:** Returns a simple "Hello World" message.
- **Usage:**
    ```bash
    curl -X GET "http://127.0.0.1:8000/"
    ```

#### 2. Image Prediction

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
    - `img_url`: URL of the image to be classified.
- **Response:**
    - `prediction_label`: Predicted label for the image.
    - `probability`: Probability score associated with the prediction.
- **Usage:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"img_url": "image_url_here"}'
    ```

### Configuration

- **Allowed Origins:** The API allows requests from specified origins, including `http://localhost:8080`, `localhost:8080`, and `*`.

### Dependencies

- [fastapi](https://fastapi.tiangolo.com/)
- [uvicorn](https://www.uvicorn.org/)
- [corsmiddleware](https://fastapi.tiangolo.com/tutorial/middleware-cors/)

Feel free to explore the API documentation and make predictions using the provided endpoints.
