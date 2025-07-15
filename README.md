# AutoML-Pipeline

An end-to-end Automated Machine Learning (AutoML) pipeline designed to streamline the process of data ingestion, model training, evaluation, and deployment. This project integrates a user-friendly interface with robust backend services to facilitate seamless machine learning workflows.

## 🚀 Features

* **Automated Data Ingestion**: Upload and preprocess datasets effortlessly through the intuitive UI.
* **Model Training & Evaluation**: Leverage automated processes to train and evaluate machine learning models.
* **RESTful API**: Interact with the pipeline programmatically via well-defined API endpoints.
* **Modular Architecture**: Clean separation between UI, API, and core AutoML logic for enhanced maintainability.

## 🗂️ Project Structure

```
AutoML-Pipeline/
├── api/             # Flask-based API endpoints
├── automl/          # Core AutoML logic and utilities
├── ui/              # Frontend interface (e.g., Streamlit or Flask templates)
├── requirements.txt # Python dependencies
└── .gitignore       # Git ignore file
```

## ⚙️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/twishapatel12/AutoML-Pipeline.git
   cd AutoML-Pipeline
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Usage

1. **Start the API Server**

   Navigate to the `api/` directory and run:

   ```bash
   python app.py
   ```

   This will start the Flask API server on `http://localhost:5000/`.

2. **Access the UI**

   Navigate to the `ui/` directory and run:

   ```bash
   streamlit run app.py
   ```

   This will launch the frontend interface, allowing you to upload datasets and initiate model training.

## 📄 API Endpoints

* `POST /upload`: Upload a new dataset.
* `POST /train`: Initiate model training.
* `GET /status`: Check the status of the training process.
* `GET /results`: Retrieve evaluation metrics and trained model details.

## 🛠️ Technologies Used

* **Frontend**: Streamlit / Flask Templates
* **Backend**: Flask
* **Machine Learning**: scikit-learn, pandas, NumPy

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
