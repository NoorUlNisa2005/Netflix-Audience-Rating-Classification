# Netflix Audience Rating Classification

A multi-class machine learning project that predicts the **audience rating category** (e.g. TV-MA, TV-14, PG-13, R) of Netflix titles using content attributes. Two models, Decision Tree and Random Forest, are trained, tuned and compared.

## Project Workflow

| Step | Description |
|------|-------------|
| 1 | Analyze rating categories |
| 2 | Prepare training datasets |
| 3 | Train classification models |
| 4 | Optimize model performance (hyperparameter tuning) |
| 5 | Evaluate prediction accuracy |

## Dataset

- **File:** `Dataset.csv`
- **Size:** 8,790 titles, 10 columns
- **Target:** `rating` (14 original categories)
- **Columns:** `show_id`, `type`, `title`, `director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`

Rating categories with fewer than 10 samples (`TV-Y7-FV`, `NC-17`, `UR`) are removed, as they are too small to train and evaluate on. This leaves 11 classes.

## Feature Engineering

| Feature | Source |
|---------|--------|
| `is_movie` | `type` (Movie = 1, TV Show = 0) |
| `duration_value` | Numeric value extracted from `duration` (minutes or seasons) |
| `release_year` | Original column |
| `year_added`, `month_added` | Parsed from `date_added` |
| `has_director` | Whether a director is listed |
| `country_*` | One-hot encoding of the primary country (top 10 + "Other") |
| Genre flags | Multi-hot encoding of the 25 most frequent genres in `listed_in` |

The data is split 80/20 into train and test sets using stratified sampling.

## Models and Tuning

- **Decision Tree:** tuned over `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Random Forest:** tuned over `n_estimators`, `max_depth`, `min_samples_leaf`
- Tuning uses `GridSearchCV` with 5-fold cross-validation and accuracy as the scoring metric.

## Results

| Model | Test Accuracy |
|-------|---------------|
| Random Forest (Tuned) | **55.6%** |
| Random Forest (Baseline) | 53.6% |
| Decision Tree (Tuned) | 51.4% |
| Decision Tree (Baseline) | 44.6% |

**Best model:** Random Forest (Tuned), with `n_estimators=200`, `max_depth=None`, `min_samples_leaf=3`.

Hyperparameter tuning improved the Decision Tree by about 7 points and the Random Forest by about 2 points over their baselines.

### Observations

- With 11 classes, random guessing gives roughly 9% accuracy, so 55.6% is a meaningful improvement.
- The model performs best on frequent classes such as `TV-MA` (F1 ≈ 0.68) and `R` (F1 ≈ 0.59).
- Performance is weak on minority and overlapping classes such as `NR`, `TV-G` and `TV-PG`.
- Ratings are closely tied to content and audience, which the available metadata only partly captures. Text features (for example, title or description) could improve results.

Exact numbers may vary slightly across library versions.

## Project Structure

```
.
├── netflix_rating_classification.py
├── Dataset.csv
├── requirements.txt
└── README.md
```

## Installation and Usage

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python netflix_rating_classification.py
```

The script prints the rating distribution, tuning results, model comparison and classification report, and displays a rating distribution chart and a confusion matrix. Tuning may take a few minutes.

## Requirements

```
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
```

## Skills Demonstrated

- Decision Trees and Random Forest
- Multi-class classification
- Feature engineering on categorical and multi-label data
- Hyperparameter tuning with cross-validation
- Classification metrics (accuracy, precision, recall, F1, confusion matrix)
