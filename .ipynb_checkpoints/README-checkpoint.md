# Recipe Recommender App

A Streamlit web app that recommends recipes based on the ingredients you have. It uses **semantic similarity** and **ingredient overlap** to find the best matches. Works for vegetarian and vegan filters, and allows controlling weights, steps, and ingredient limits.

---

## Features

- Enter the ingredients you have and get recommended recipes.  
- Filter recipes by:
  - Maximum number of steps
  - Maximum number of ingredients
  - Vegetarian / Vegan
- Control semantic match vs ingredient overlap weighting.  
- Highlight matched ingredients in results.  
- Download recommended recipes as CSV.

---

## Installation

**Clone the repository**
```bash
git clone https://github.com/rohan-crypto/food-recipe-recommender.git
cd food-recipe-recommender

## Create a virtual environment

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

## Install dependencies

pip install -r requirements.txt

## Running Locally

streamlit run app.py

