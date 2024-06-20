<br />
<div align="center">
  <h3 align="center">Customer Churn Prediction</h3>

  <p align="center">
    An end-to-end machine learning pipeline to predict customer churn, including training, prediction, and deployment to AWS EC2 with Docker and GitHub Actions.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#tech-stack">Tech Stack</a></li>
      </ul>
    </li>
    <li>
      <a href="#how-to-run">How to Run</a>
    </li>
    <li>
      <a href="#training-pipeline">Training Pipeline</a>
    </li>
    <li>
      <a href="#prediction-pipeline">Prediction Pipeline</a>
    </li>
    <li>
      <a href="#deployment">Deployment</a>
    </li>
    <li>
      <a href="#workflow">Workflow</a>
    </li>
    <li>
      <a href="#aws-cicd-deployment-with-github-actions">AWS CI/CD Deployment with GitHub Actions</a>
    </li>
  </ol>
</details>

## About The Project

This project is designed to accurately predict customer churn. The pipeline involves data ingestion, validation, transformation, model training, evaluation, and deployment.

### Tech Stack

- **Programming Language**: Python
- **Database**: MongoDB
- **Containerization**: Docker
- **Cloud Services**: 
  - AWS S3: Model storage
  - AWS EC2: Hosting the application
  - AWS ECR: Docker image repository
- **CI/CD**: GitHub Actions
- **Web Application**: HTML, CSS

## How to Run

Instructions to set up your local environment for running the project.

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## Usage

Instructions on how to use the project.

```bash
# Run the application
python app.py
```

## Training Pipeline

1. **Data Ingestion**: 
   - Ingest data from MongoDB.
   - Split the data into train and test sets.
   - Export the data from MongoDB to CSV files for further processing.

2. **Data Validation**:
   - Validate the structure and format of the ingested data.
   - Check for data drift using Evidently to ensure the data consistency over time.

3. **Data Transformation**:
   - Transform raw data into a suitable format for model building.
   - Apply transformations like One-Hot Encoding, Ordinal Encoding, and Scaling.
   - Handle imbalanced data using techniques like SMOTEENN.

4. **Model Training**:
   - Train multiple machine learning models such as K-Neighbors, Random Forest, XGBoost, and CatBoost.
   - Perform hyperparameter tuning using GridSearchCV to find the best model with optimal parameters.

5. **Model Evaluation**:
   - Evaluate the best model from the training pipeline.
   - Compare with the production model stored in Amazon S3.
   - Use the best-performing model for predictions on test data.

6. **Model Pusher**:
   - Store the final best-trained model in AWS S3 for making predictions.
   - Upload model artifacts to S3 for future use.

7. **Training Pipeline Execution**:
   - Execute the entire training pipeline to process data, train, evaluate, and deploy the model.

## Prediction Pipeline

- Ingest new or unseen data from users or MongoDB.
- Transform the new data using the preprocessing steps from the training pipeline.
- Make predictions using the best-trained model stored in AWS S3.

## Deployment

1. **Containerize the Application**:
   - Use Docker to containerize the application for easy deployment and scalability.
   - Store the Docker image in the AWS ECR repository.

2. **Set up AWS EC2 Instance**:
   - Host the deployed application on an AWS EC2 instance.
   - Pull the Docker image from AWS ECR and run the application on EC2.

3. **Automate Deployment with GitHub Actions**:
   - Use GitHub Actions to automate the deployment workflow.
   - On each code push, retrain the model, build the Docker image, push it to AWS ECR, pull the image to EC2, and run the application.

4. **Web Application**:
   - Build a basic web app using HTML and CSS to expose the model's prediction functionality.

## Workflow

1. constant
2. config_entity
3. artifact_entity
4. component
5. pipeline
6. app.py / demo.py

### Export the Environment Variables

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>...."
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```

## AWS CI/CD Deployment with GitHub Actions

1. **Login to AWS Console.**

2. **Create IAM User for Deployment**

   - **With specific access:**
     1. EC2 access: It is a virtual machine.
     2. ECR: Elastic Container Registry to save your Docker image in AWS.

   - **Description: About the deployment:**
     1. Build Docker image of the source code.
     2. Push your Docker image to ECR.
     3. Launch Your EC2.
     4. Pull Your image from ECR in EC2.
     5. Launch your Docker image in EC2.

   - **Policy:**
     1. AmazonEC2ContainerRegistryFullAccess
     2. AmazonEC2FullAccess

3. **Create ECR Repo to Store/Save Docker Image**
   - Save the URI: `<your URI>`

4. **Create EC2 Machine (Ubuntu)**

5. **Open EC2 and Install Docker in EC2 Machine:**
   
   - **Optional:**

     ```bash
     sudo apt-get update -y
     sudo apt-get upgrade
     ```

   - **Required:**

     ```bash
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     sudo usermod -aG docker ubuntu
     newgrp docker
     ```

6. **Configure EC2 as Self-Hosted Runner:**
   - Go to settings > actions > runner > new self-hosted runner > choose OS > then run commands one by one.

7. **Setup GitHub Secrets:**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION`
   - `ECR_REPO`
