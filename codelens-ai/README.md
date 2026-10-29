# CodeLens AI — ML-based Python Logic-Error Detection & AST Analysis

A full-stack website that detects **logical errors** in Python `while`-loop
programs, auto-corrects them, and classifies the error location with machine
learning over the code's **Abstract Syntax Tree (AST)**.

Built strictly on the base paper:

> Leyton-Yela, Mora, Insuasti & Barco-Jiménez (2026),
> **"Machine Learning Models for Automatic Classification of Programming Tasks
> in Python"**, IEEE Revista Iberoamericana de Tecnologías del Aprendizaje
> (RITA), vol. 21.

## Features

1. **Login / Sign-up** (per-user accounts, sessions)
2. **Home** — project overview + the Gries S/T/E state model
3. **Analyze Code**
   - Paste Python code → **Upload & Check Logic**
   - Rule-based AST engine reports `CORRECT` / `WRONG`, with every error tagged
     by Gries state — **S** (Initial state), **T** (Transformation),
     **E** (Final state) — line number, explanation, severity
   - **Auto-corrected code** is displayed
   - **Run ML AST Analysis** → the corrected code is classified into the
     paper's 7 Gries labels by a Voting(RF+MLP) ensemble trained on the
     3,000-task dataset
   - **matplotlib charts**: per-class probability bar chart, AST feature
     profile across S/T/E, plus the full AST tree view and feature vector
   - Every analysis is **saved to history**
4. **Dashboard** — paper-style matplotlib charts: model comparison bar chart
   (Accuracy/AUC/MCC for KNN, SVM, RF, MLP, Voting, Stacking), dataset label
   distribution, confusion matrix (Fig. 12), top AST feature importances
5. **Dataset** — searchable/filterable table of the 3,000 generated records
   (Description, Solution, Feedback, Label) + CSV download
6. **Profile** — account stats and full analysis history

## Base-paper methodology reproduced

| Paper element | Implementation |
|---|---|
| Gries states S / T / E | `WhileLoopFinder` partitions the AST into statements before the loop (S), loop body (T), loop condition + statements after (E) |
| 7 error labels ys, ye, yt, yst, yse, yet, yste | `dataset_builder.py` mutates exactly the S / T / E states (Prompt-1 style) |
| 3,000 loop tasks, 29/29/29/4/3% distribution | `engine/dataset_builder.py` → `data/gries_loop_tasks.csv` |
| Dataset sources [25][26][27] | Project CodeNet / Codeforwin / PYnative-style beginner while-loop exercises |
| Seed 2023, stratified 80/20 split | `engine/ml_engine.py` |
| Models KNN-15 Manhattan, SVM-RBF (γ=0.000379, C=233.57), RF-70/depth-15, MLP [100,100,100] adaptive, Voting, Stacking | scikit-learn pipelines in `ml_engine.py` |
| Metrics Accuracy / AUC / MCC | paper Table I comparison matrix |
| Bar charts (Fig. 7/9/10) + confusion matrix (Fig. 12) | matplotlib dark-theme charts |

**No external data is downloaded** — the dataset is generated locally in the
exact structure described by the paper (CSV: Description, Solution, Feedback,
Label), using only the repositories named in the paper's reference list.

## Run

```bash
cd codelens-ai
pip install flask pandas scikit-learn matplotlib pypdf
python -m engine.ml_engine     # trains models once (~25 s) → data/model_bundle.pkl
python app.py                  # http://localhost:5000
```

Create an account on the login page and click **Analyze Code**.
