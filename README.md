# Machine Learning Project with SQL Backend

## Overview
This project aims to build and evaluate various machine learning models using data stored in an SQL database.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

# Project Description

This project is focused on training various machine learning models on a dataset stored in an SQLite database. The models include Logistic Regression, Random Forest, AdaBoost, GradientBoosting, SVM with RBF kernel, and Naive Bayes. The project also includes data preprocessing, feature selection, and model evaluation (AUC, ROC, F1-Score).

## How to Run
sqlite3 data/project.db < data/init.sql
sqlite3 data/project.db < data/populate.sql
python main.py
