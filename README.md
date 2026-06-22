# Movie Recommendation System

An end-to-end machine learning project designed to analyze movie data and recommend films to users based on content similarity. This repository contains the complete source code, a comprehensive project report, and a presentation deck summarizing the system's lifecycle.

## 📁 Repository Structure

This repository includes the following key files:
* **`movie_recomendation_code.ipynb`**: The primary Jupyter Notebook containing data preprocessing, exploratory data analysis (EDA), feature engineering, and the core recommendation engine logic.
* **`Movie_Recommendation_Report.docx`**: A detailed project documentation report covering the background methodology, algorithms, and results.
* **`Movie-Recommendation-System.pptx`**: A presentation deck used to summarize the project objectives, architecture, and findings.

---

## 🛠️ Features & Methodology

* **Data Preprocessing:** Handles missing values, cleans textual attributes, and extracts relevant information from complex metadata fields.
* **Feature Engineering:** Combines key movie elements (such as genres, keywords, overview, and cast/crew tags) into consolidated text vectors.
* **Vectorization:** Converts the textual metadata into numerical feature vectors using **TF-IDF Vectorizer** or **CountVectorizer**.
* **Similarity Computation:** Utilizes **Cosine Similarity** to compute distance matrices between movies, surfacing the top-N most relevant recommendations based on user input.

---

## 🚀 Getting Started

To run the notebook locally and explore the recommendations, follow these steps:

### Prerequisites 
Ensure you have Python installed alongside the following required data science libraries:

pip install pandas numpy scikit-learn notebook

### Setup and Execution

1. **Clone the repository:**
   
   git clone [https://github.com/Dathrija05/movie-recomendation.git](https://github.com/Dathrija05/movie-recomendation.git)
   
2. **Navigate into the project directory:**
   
   cd movie-recomendation
   
3. **Launch Jupyter Notebook:**

   jupyter notebook

4. **Run the project:**
   
   Open movie_recomendation_code.ipynb and run the cells sequentially to see the data processing and recommendation outputs.
   
   
