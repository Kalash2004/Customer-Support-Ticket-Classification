# 📨 Customer Support Ticket Classification with NLP & XGBoost
## Table of Contents

1.  Project Overview

2. Dataset

3. Key Features

4. Project Architecture

5. Getting Started

6. Prerequisites

7. Installation

8. Usage

8. Results

10.Contributing



## 🚩 Project Overview
Customer Support Ticket Classification leverages Natural Language Processing (NLP) and Machine Learning (XGBoost) to automatically categorize customer support tickets by their issues, type, or urgency.

Goal: Replace slow and error-prone manual ticket triaging with fast, accurate, and scalable automated classification, enabling customer support teams to route issues to the correct departments quickly.

Why?
Manual ticket assignment takes hours, bottlenecks productivity, and leads to misrouted queries.

ML-driven automation improves response times, accuracy, and customer satisfaction.

## 📊 Dataset
This project utilizes multiple real-world and synthetic datasets for customer service ticket processing, including:

customer_support_tickets.csv:
Contains fields like Ticket ID, Customer Info, Product, Ticket Type, Subject, Description, Status, Priority, Channel, Timestamps, and Satisfaction Ratings.

Bitext_Sample_Customer_Service_Training_Dataset.csv:
Contains utterances and labeled intents for categorization.

twcs.csv:
Conversations on Twitter customer care (optional for dialogue/intent modeling).

Sample Data Fields
Field	Example Value
Ticket ID	1
Customer Name	Marisa Obrien
Product Purchased	GoPro Hero
Ticket Description	I'm having an issue with ...
Ticket Priority	Critical
Ticket Status	Pending Customer Response
Date of Purchase	2021-03-22
Resolution	...
Channel	Social media, Chat, etc.
Note: Only sample/synthetic data is included in this repo for privacy. Replace with your real data accordingly.

## ✨ Key Features
End-to-End NLP Pipeline: Data cleaning, feature extraction, and text preprocessing.

Advanced ML Algorithms: Uses XGBoost for robust ticket classification.

Rich Feature Engineering: Including text-derived metrics (word count, urgency flags, category scores).

Model Evaluation & Visualization: Confusion matrix, accuracy, and detailed performance metrics.

Extensible Codebase: Easy to adapt to new ticket datasets and categories.

## 🏗️ Project Architecture
Data Ingestion: Read and clean customer ticket datasets

Feature Engineering: Extract key features (urgency, intent, topic)

Text Preprocessing: Tokenization, stop-word removal, and embedding (TF-IDF or Count Vectorizer)

Model Training: XGBoost classifier (tuned for multi-class or binary classification)

Evaluation: Reports accuracy, confusion matrix, and business impact

Prediction: Assigns new tickets to the correct department in real time

## 🚀 Getting Started
Prerequisites
Python 3.7+

pip (Python package manager)

Required Libraries

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

You can install them via:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/Ticket-Classification-ML.git
cd Ticket-Classification-ML
```
Install dependencies:
```bash
pip install -r requirements.txt
```
(Optional) Add your data:
Place your ticket data in the data/ directory, formatted as shown above.

Usage
Run Data Exploration & Feature Engineering:

```bash
jupyter notebook notebooks/data_exploration.ipynb
(or use your main Python script as per the project structure)
```
Train the Classifier:

```bash
python main.py
```
Predict on New Tickets:
Use the built/served model to predict categories for new ticket inputs.
## 📈 Results
Model: XGBoost Classifier (best hyperparameters: max_depth=4, learning_rate=0.1, n_estimators=100)

Accuracy: Achieved >95% on test data.

Impact: Reduced manual triaging time from hours to seconds, decreased misclassification rate, and improved customer satisfaction.

## 🙌 Contributing
Contributions are welcome! To contribute, Fork this repo

HAPPY LEARNING


Create a new branch (git checkout -b feature-branch)

Commit your changes (git commit -m "Add feature")

Push to the branch (git push origin feature-branch)

Create a pull request

Please open issues for any bugs or feature suggestions.
