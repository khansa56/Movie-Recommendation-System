# 🎬 Movie Recommendation System (Content-Based)

An intelligent content-based movie recommendation system built in Python using Machine Learning and NLP techniques. The system calculates text similarity between movie genres and descriptions using **TF-IDF Vectorization** and **Cosine Similarity** to recommend similar movies based on user preferences.

---

## ✨ Features

* **Content-Based Filtering:** Recommends movies based on plot keywords, categories, and genres.
* **TF-IDF Vectorization:** Converts unstructured genre text into meaningful mathematical feature vectors.
* **Cosine Similarity Matching:** Measures the spatial angle between movie vectors to find the closest thematic matches.
* **Interactive CLI Interface:** User-friendly command-line prompt for instant recommendations.

---

## 🛠️ Tech Stack & Dependencies

| Library | Purpose |
| :--- | :--- |
| **Python 3.x** | Core programming language |
| **pandas** | Data manipulation and dataset handling |
| **scikit-learn** | Machine learning utilities (`TfidfVectorizer` & `cosine_similarity`) |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can install the required packages using `pip`:

```bash
pip install pandas scikit-learn
