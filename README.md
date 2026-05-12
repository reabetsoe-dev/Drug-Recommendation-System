# AIDrugReview

AIDrugReview is a drug review analysis system that predicts a patient's likely condition from review text and recommends highly rated medications for that condition. The system focuses on three target conditions:

- Depression
- High Blood Pressure
- Type 2 Diabetes

The project uses the Drugs.com review dataset, trains a text classification model, serves predictions through FastAPI services, and displays results in a React dashboard.

## Project Folders

```text
AIDrugReview/
- data/              Raw Drugs.com CSV files used for training and recommendation
- training/          Model creation, cleaning, feature extraction, tuning, and plots
- models/            Saved model artifacts used during inference
- ai_service/        FastAPI inference and recommendation service
- backend/           FastAPI gateway used by the frontend
- frontend/          React dashboard source, visual UI, PDF report download, image assets
- frontend/assets/   Landing page images, background image, and logo assets
- shared/            Shared paths, ports, and configuration
- nltk_data/         Local NLTK tokenizer and stopword resources
```

## Main Dataset Features

### DrugName, categorical

In the raw CSV files this column is stored as `drugName`.

How it is achieved:

- The raw drug names are loaded from `data/drugsComTrain_raw.csv` and `data/drugsComTest_raw.csv`.
- The training pipeline in `training/main.py` keeps `drugName` as a categorical identifier rather than converting it into model text features.
- During inference, `ai_service/model_api.py` uses `drugName` for recommendation ranking.
- The `recommend_drugs()` function groups reviews by `drugName`, then calculates:
  - average rating
  - number of reviews
  - average useful count
- The grouped drug ranking is sorted by high average rating and review count.
- The frontend displays recommended drug names in medication cards and visual rating charts.

Purpose:

`drugName` makes it possible to compare drugs within the predicted condition and recommend the strongest options based on patient feedback.

### Condition, categorical

In the raw CSV files this column is stored as `condition`.

How it is achieved:

- `training/main.py` loads the condition column during dataset preparation.
- The `clean_data()` function removes rows where `condition`, `review`, or `rating` is missing.
- The `filter_conditions()` function keeps only reviews related to:
  - Depression
  - High Blood Pressure
  - Type 2 Diabetes
- Similar source labels are normalized into the three final class names.
- The `LabelEncoder` in `training/main.py` converts the condition labels into numeric classes for model training.
- The trained label encoder is saved to `models/label_encoder.pkl`.
- During prediction, `ai_service/model_api.py` converts the numeric model output back into the readable condition name.

Purpose:

`condition` is the target label for classification. It teaches the model which condition a review belongs to and is also used to filter drug recommendations.

### Review, text

In the raw CSV files this column is stored as `review`.

How it is achieved:

- `training/main.py` cleans review text with the `clean_text()` function.
- Cleaning includes:
  - converting text to lowercase
  - removing punctuation
  - removing numbers
  - normalizing spaces
  - tokenizing text with NLTK
  - removing English stopwords
  - removing very short tokens
- The cleaned text is stored as `cleaned_review`.
- `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` converts cleaned reviews into numerical text features.
- The vectorizer learns unigram and bigram patterns from patient language.
- The vectorizer is saved to `models/vectorizer.pkl`.
- The selected trained classifier is saved to `models/model.pkl`.
- At runtime, the React frontend collects user review text, sends it to `backend/main.py`, and the backend forwards it to `ai_service/model_api.py`.
- `ai_service/model_api.py` cleans the new review text with the same style of preprocessing, transforms it with the saved TF-IDF vectorizer, and predicts the likely condition.
- The AI service also applies a lightweight word-based sentiment check to classify the review tone as positive, neutral, or negative.

Purpose:

`review` is the main predictive feature. It lets the model learn language patterns associated with each condition and produce real-time predictions from new patient text.

### Rating, numerical

In the raw CSV files this column is stored as `rating`.

How it is achieved:

- `training/main.py` validates ratings by keeping only values from 1 to 10.
- Rating distributions and boxplots are generated during model creation and saved under `artifacts/plots/`.
- `ai_service/model_api.py` uses ratings in two ways:
  - `estimate_rating_prediction()` calculates the average rating for the predicted condition and adjusts it using model confidence.
  - `recommend_drugs()` groups reviews by `drugName` and calculates the average rating for each drug.
- The frontend displays rating-based recommendation cards and a rating comparison chart.
- The PDF report generated in the frontend includes the predicted condition, confidence, estimated rating, and recommended medications.

Purpose:

`rating` measures patient satisfaction. It helps the system rank drugs by effectiveness and provide evidence-based recommendations.

## Model Creation Flow

The model pipeline is implemented in `training/main.py`.

1. Load datasets from `data/`.
2. Clean missing values, duplicates, invalid ratings, and dates.
3. Filter the dataset to the three target conditions.
4. Clean review text and create `cleaned_review`.
5. Generate exploratory plots in `artifacts/plots/`.
6. Convert condition labels with `LabelEncoder`.
7. Convert text into TF-IDF vectors with 5,000 maximum features.
8. Train and compare:
   - Logistic Regression
   - Multinomial Naive Bayes
9. Tune models with `GridSearchCV`.
10. Save final artifacts to `models/`:
   - `model.pkl`
   - `vectorizer.pkl`
   - `label_encoder.pkl`

Run training:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview
.\.venv\Scripts\python training\main.py
```

## Runtime Architecture

### AI service: `ai_service/`

`ai_service/model_api.py` loads the saved model, vectorizer, label encoder, and training dataset.

Responsibilities:

- health check at `/health`
- prediction and recommendation at `/analyze`
- text preprocessing
- condition prediction
- confidence calculation
- sentiment estimate
- rating estimate
- drug recommendation ranking

### Backend API: `backend/`

`backend/main.py` is a gateway between the frontend and AI service.

Responsibilities:

- health check at `/health`
- frontend prediction endpoint at `/predict`
- forwarding requests to the AI service
- returning clean JSON responses to the React UI

### Frontend dashboard: `frontend/`

The frontend is a Vite React user interface.

Responsibilities:

- landing page with logo and background images
- condition prediction text box
- known-condition recommendation mode
- backend API calls
- result display
- confidence charts
- recommended medication cards
- rating comparison chart
- downloadable PDF analysis report

### Assignment Evidence

After running `training/main.py`, the project writes presentation-ready evidence to `artifacts/`:

- `data_quality_summary.csv`
- `feature_tuning_results.csv`
- `model_results.csv`
- `tuned_model_results.csv`
- classification report CSV files
- plots for condition distribution, rating distribution, useful counts, date trends, feature tuning, model comparison, and confusion matrices

The presentation deck is in `presentation/AI_Drug_Review_Presentation.pptx`.

## Running the Project

The system runs as three local services:

- AI service on port `5000`
- Backend API on port `8000`
- React frontend on port `8501`

### First-time setup

From the project root:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview
```

Create a virtual environment if `.venv/` does not already exist:

```cmd
python -m venv .venv
```

Install Python dependencies:

```cmd
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Install Node dependencies for the React frontend:

```cmd
npm install
```

The project expects these trained model files to exist in `models/`:

```text
models/model.pkl
models/vectorizer.pkl
models/label_encoder.pkl
```

If they are missing, run the training pipeline:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview
.\.venv\Scripts\python training\main.py
```

### Start the AI service

Open Terminal 1:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview\ai_service
..\.venv\Scripts\python -m uvicorn model_api:app --host 127.0.0.1 --port 5000
```

Check it in the browser:

```text
http://127.0.0.1:5000
http://127.0.0.1:5000/health
http://127.0.0.1:5000/docs
```

### Start the backend API

Open Terminal 2:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview\backend
..\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Check it in the browser:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### Start the frontend

Open Terminal 3:

```cmd
cd /d C:\Users\reabe\Desktop\AIDrugReview
npm run dev
```

Then open:

```text
http://localhost:8501
```

The React dev server proxies `/api` requests to `http://127.0.0.1:8000`, so the browser can use the existing backend routes.

### Quick health checks

Use these PowerShell commands to confirm the services are responding:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:8000/health
```

The backend health response should include the nested AI service status. If the backend is running but the AI service is not, predictions will fail until the AI service is started.

### Stop the services

If the services are running in visible terminals, press `Ctrl+C` in each terminal.

To find processes using the ports:

```powershell
netstat -ano | findstr ":5000 :8000 :8501"
```

To stop a known process ID:

```powershell
Stop-Process -Id <PID> -Force
```

### Service URLs

- Frontend: `http://localhost:8501`
- Backend: `http://127.0.0.1:8000`
- Backend health: `http://127.0.0.1:8000/health`
- Backend docs: `http://127.0.0.1:8000/docs`
- AI service: `http://127.0.0.1:5000`
- AI service health: `http://127.0.0.1:5000/health`
- AI service docs: `http://127.0.0.1:5000/docs`

## Summary

The four core features work together as follows:

- `drugName` identifies and ranks medications.
- `condition` provides the classification target and recommendation filter.
- `review` provides the text used for machine learning prediction.
- `rating` provides patient satisfaction evidence for ranking and outcome estimation.

Together, these features allow the system to predict a medical condition from patient language and recommend drugs using real patient review ratings.
