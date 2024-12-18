# Memesense

## Overview

This project focuses on developing a multimodal machine learning model to classify memes based on their content and categories. By leveraging both image and text data, the project demonstrates the integration of computer vision and natural language processing techniques for effective meme classification.

## Objectives

- Build a robust dataset of memes with appropriate labels.

- Implement a multimodal machine learning model combining image and text data.

- Evaluate the model using standard performance metrics.

- Deploy the final model for practical use.


## Dataset

- **Source**: Memotion Dataset (7K images and labels).

- **Size**: 7,300 labeled memes. (Additional memes were added manually.)

- **Categories**: Positive, Negative.

- **Preprocessing**:

    - Removed duplicates and empty rows.

    - Cleaned text data by lowercasing and removing special characters.

    - Resized and normalized images to 224x224.

    - Verified the existence of corresponding images for all records.


## Methodology

### Data Preprocessing

1. **Text Processing**:

    - Used `BertTokenizer` to tokenize text data with a maximum length of 50 tokens.

    - Encoded labels into numerical format using `LabelEncoder`.

2. **Image Processing**:

    - Resized images to 224x224 and normalized pixel values.

    - Ensured all images were in RGB format.

3. **Data Splitting**:

    - Split the data into training (80%) and testing (20%) sets.

    - Ensured alignment between text and image data.


### Model Architecture

1. **Image Model**:

    - Utilized ResNet50 pre-trained on ImageNet as the base model.

    - Added a GlobalAveragePooling2D layer for feature extraction.

2. **Text Model**:

    - Used BERT from Hugging Face for text feature extraction.

    - Token IDs and attention masks were used as input.

3. **Multimodal Fusion**:

    - Combined image and text features using a `Concatenate` layer.

    - Added fully connected layers with dropout for regularization.

4. **Output Layer**:

    - Used a softmax activation function to classify memes into the predefined categories.

5. **Class Balancing**:

    - Computed class weights to handle imbalanced data during training.


### Training

- Optimized with Adam optimizer and a learning rate of 0.0001.

- Used `EarlyStopping` to avoid overfitting, with patience set to 3 epochs.

- Trained for up to 20 epochs with a batch size of 32.


### Evaluation

- Evaluated the model using accuracy, F1-score, and a confusion matrix.

- Visualized results using Matplotlib and Seaborn.


## Results

### Metrics

- **Test Accuracy**: 53% - Model performance is moderate and highlights opportunities for further optimization. It is worth noting that research on meme classification often reports accuracies around 60%, reflecting the inherent challenges in combining image and text data for such tasks.

- **Classification Report**:

    - Precision, recall, and F1-scores were calculated for all categories.


### Visualizations

1. **Confusion Matrix**:

    - Displayed using a heatmap for detailed error analysis.

2. **Loss and Accuracy Curves**:

    - Showed the model's performance during training and validation.

3. **Predictions**:

    - Included examples of model predictions with corresponding true labels and visualizations.


### Challenges

- Balancing the dataset (The dataset has a class imbalance, with a higher proportion of positive labels.).

- Ensuring robust integration of image and text features.


## Deployment

The deployment of this project involves multiple steps:

1. **API Creation**:

    - An API was built using FastAPI and uvicorn to serve the model for predictions.

    - The API accepts image and text inputs and returns the predicted category.

2. **Frontend with Streamlit**:

    - A user-friendly interface was developed using Streamlit.

    - Users can upload memes and view predictions directly on the web interface.

    - The Streamlit app allows users to upload a meme image, from which the system extracts both image and text features. These features are processed by the model to generate a category prediction.


3. **Local Testing**:

    - The API and frontend were tested locally to ensure seamless interaction and correct predictions.

4. **Cloud Deployment**:

    - The project is being deployed on Google Cloud Platform (GCP).

    - A Docker image was created to containerize the application.

    - Deployment includes both the API and frontend for a fully integrated solution.

5. **Streamlit Cloud**:

    - The project frontend is being deployed on Streamlit Cloud.

    - The deployment integrates with the AVColman/memesense_interface repository.



## Tools and Technologies

- **Programming Languages**: Python

- **Libraries and Frameworks**:

    - TensorFlow/Keras for model development.

    - Hugging Face Transformers for BERT.

    - OpenCV for image preprocessing.

    - Matplotlib and Seaborn for visualizations.

    - Scikit-learn for evaluation metrics.

    - FastAPI and uvicorn for API development.

    - Streamlit for frontend development.

    - Docker for containerization.

    - Streamlit Cloud for frontend hosting


## Installation

To run this project locally:

1. Clone the repository:

    ```
    git clone https://github.com/MonicaVenzor/memesense
    ```

2. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

3. Run the API and Streamlit app:

    ```
    python api.py
    streamlit run app.py
    ```

4. To build and run the Docker container locally:

    ```
    docker build -t [REGION]/[PROJECT_ID]/[REPOSITORY]/[IMAGE_NAME]:[TAG] .
    docker run -e PORT=8000 -p 8000:8000 [REGION]/[PROJECT_ID]/[REPOSITORY]/[IMAGE_NAME]:[TAG]
    docker push [REGION]/[PROJECT_ID]/[REPOSITORY]/[IMAGE_NAME]:[TAG]
    ```

5. Cloud deployment:

    ```
    gcloud run deploy [SERVICE_NAME] \
    --image [REGION]/[PROJECT_ID]/[REPOSITORY]/[IMAGE_NAME]:[TAG] \
    --platform managed \
    --region [REGION]\
    --allow-unauthenticated \
    --memory 4Gi

    ```

6. Streamlit Cloud deployment:

    ```
    https://streamlit.io/cloud
    Follow the instructions provided in the Deployment section.
    ```

## Future Work

- Add additional meme categories, such as humorous, satirical, or political memes, to improve model generalization.

- Experiment with advanced architectures, such as Vision Transformers.

- Optimize the deployment for scalability and performance.


## Contributors

- Monica Venzor
- Alina Colman
- Gerardo Vargas

## Acknowledgments

- Memotion Dataset for providing labeled data from Kaggle.

- TensorFlow and Hugging Face for pre-trained models and tools.

- Open-source tools and libraries that supported this project.
