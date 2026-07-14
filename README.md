# 🎬 TMDB 5000 Movie Recommendation System

A comprehensive Data Science and Machine Learning project focused on building multi-faceted recommendation workflows using the popular TMDB 5000 Movies dataset[cite: 4]. The system leverages statistical metrics, IMDB's Weighted Rating formula, and hybrid scaling approaches to recommend the top most critically and popularly engaging movies.

---

## 🎯 Project Overview & Objective
Modern content streaming platforms rely heavily on recommendation algorithms to increase user engagement. The goal of this project is to analyze raw movie metadata, handle textual descriptors, and engineer statistical scores to build:
1. **A Demographic/Content-Weighted Recommendation Engine**: Based on IMDB's mathematically corrected Bayesian average system[cite: 4].
2. **A Popularity-Based Recommendation Engine**: Ranking system based on user-interaction quantities.
3. **A Hybrid Recommendation Engine**: A normalized combination giving equal mathematical weights (50-50 split) to both critical reception and popularity trends.

---

## 💾 Dataset Schema & Architecture
The project ingests two relational source files containing extensive movie characteristics[cite: 4]:
* **`tmdb_5000_credits.csv`**: Contains production details (`movie_id`, `title`, `cast`, `crew`)[cite: 4].
* **`tmdb_5000_movies.csv`**: Contains core financial and metadata values (`budget`, `genres`, `id`, `keywords`, `popularity`, `release_date`, `revenue`, `vote_average`, `vote_count`, etc.)[cite: 4].

### Features Handled:
* **id / movie_id**: Unique key identifier used to merge both tabular structures[cite: 4].
* **vote_average**: The true mathematical average of user-submitted scores[cite: 4].
* **vote_count**: The volume of total ratings submitted for a given record[cite: 4].
* **popularity**: A continuous numerical measure tracking trending user engagement[cite: 4].

---

## ⚙️ Step-by-Step Project Workflow

The implementation lifecycle follows a structured engineering methodology from raw files to refined rankings:

### 1. High-Level Overview & Ingestion
* Setup the system environment by loading standard analysis frameworks (`pandas`, `numpy`, `matplotlib`, `seaborn`)[cite: 4].
* Read both `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv` into individual dataframes[cite: 4].

### 2. Data Cleaning & Preparation
* Renamed `movie_id` to `id` inside the credits dataframe to align the primary relational key[cite: 4].
* Merged both datasets using internal Pandas logic on the shared `id` key column, creating an aggregated shape of `(4803, 23)`[cite: 4].
* Performed feature pruning by dropping uninformative variables (`homepage`, `title_x`, `title_y`, `status`, `production_countries`) to keep the data structured and performant[cite: 4].

### 3. Demographic & Weighted Average Recommendation Engine
* Implemented the **IMDB Weighted Rating (WR)** formula to avoid the bias of high scores with very few votes:
  $$\text{Weighted Rating } (W) = \left(\frac{v}{v+m} \cdot R\right) + \left(\frac{m}{v+m} \cdot C\right)$$
  Where:
  * $v$ = Number of votes for the movie (`vote_count`)[cite: 4]
  * $m$ = Minimum votes required to be listed in the chart (Set to the 90th percentile of total votes $\approx 1838.4$)
  * $R$ = Average rating of the movie (`vote_average`)[cite: 4]
  * $C$ = The mean vote across the entire report dataset ($\approx 6.09$)[cite: 4]
* Filtered the data frame to strictly retain movies with `vote_count` $\ge m$ (yielding 481 qualified records).
* Calculated the `weighted_avg` column dynamically across the qualified subset.

### 4. Popularity-Based Recommendation Engine
* Analyzed the distribution of movie trend metrics and sorted the data by the `popularity` attribute to see which blockbusters hold the highest historical user traction independently.

### 5. Hybrid Recommendation Model (Weighted Average + Popularity)
* Because `weighted_avg` (scaled out of 10) and `popularity` (scaled up to hundreds) have entirely different mathematical ranges, a direct combination would heavily skew the model towards popularity.
* Applied `MinMaxScaler` from `sklearn.preprocessing` to transform both features onto a uniform scale between `0` and `1`.
* Engineered a custom feature `score_mix` combining both scaled metrics via a balanced 50% split:
  $$\text{score\_mix} = (\text{weighted\_avg\_scaled} \times 0.5) + (\text{popularity\_scaled} \times 0.5)$$
* Sorted the movie ranking dynamically to extract the final combined top recommended list.

---

## 📈 Top Project Recommendations (`score_mix`)
Based on the final hybrid model, the top 5 most highly recommended movies are:
1. **Interstellar** (score_mix: 0.870)
2. **Minions** (score_mix: 0.699)
3. **Guardians of the Galaxy** (score_mix: 0.697)
4. **Deadpool** (score_mix: 0.647)
5. **The Dark Knight** (score_mix: 0.582)

---

## 🚀 Local Installation & Quick Start

### 1. Install Dependencies
Make sure you have python installed, then run the terminal command to add the necessary data science libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
