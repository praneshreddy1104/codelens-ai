"""
CodeLens AI - Flask web application.

ML-powered Python logic-error detector & AST/Gries-state analyser, built on:
  Leyton-Yela, Mora, Insuasti, Barco-Jimenez (2026),
  "Machine Learning Models for Automatic Classification of Programming Tasks
   in Python", IEEE RITA.
"""
import hashlib
import json
import os
import functools
from datetime import datetime

import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, send_file)

from engine.gries import analyze_logic, extract_features, ast_tree_text, FEATURE_NAMES
from engine import ml_engine

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
HIST_FILE = os.path.join(DATA_DIR, "history.json")
DATA_CSV = os.path.join(DATA_DIR, "gries_loop_tasks.csv")

app = Flask(__name__)
app.secret_key = "codelens-ai-gries-seed2023"

# ---------------------------------------------------------------------------
# tiny JSON "database"
# ---------------------------------------------------------------------------

def _load(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def hash_pw(pw):
    return hashlib.sha256(("codelens$" + pw).encode()).hexdigest()


def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    for u in _load(USERS_FILE, []):
        if u["id"] == uid:
            return u
    return None


def login_required(f):
    @functools.wraps(f)
    def wrap(*a, **kw):
        if not current_user():
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrap


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("home") if current_user() else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        action = request.form.get("action", "login")
        name = request.form.get("username", "").strip()
        pw = request.form.get("password", "")
        users = _load(USERS_FILE, [])
        if action == "register":
            if not name or not pw:
                err = "Username and password are required."
            elif any(u["username"] == name for u in users):
                err = "Username already exists — please log in."
            else:
                uid = hashlib.md5(name.encode()).hexdigest()[:10]
                users.append({"id": uid, "username": name,
                              "password": hash_pw(pw),
                              "created": datetime.now().strftime("%Y-%m-%d %H:%M")})
                _save(USERS_FILE, users)
                session["uid"] = uid
                return redirect(url_for("home"))
        else:
            match = next((u for u in users if u["username"] == name), None)
            if match and match["password"] == hash_pw(pw):
                session["uid"] = match["id"]
                return redirect(url_for("home"))
            err = "Invalid username or password."
    return render_template("login.html", err=err)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html", user=current_user())


@app.route("/analyze")
@login_required
def analyze_page():
    return render_template("analyze.html", user=current_user())


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user())


@app.route("/dataset")
@login_required
def dataset_page():
    return render_template("dataset.html", user=current_user())


@app.route("/profile")
@login_required
def profile():
    hist = [h for h in _load(HIST_FILE, []) if h["uid"] == session["uid"]]
    clean = sum(1 for h in hist if h["verdict"] == "correct")
    wrong = len(hist) - clean
    return render_template("profile.html", user=current_user(),
                           total=len(hist), clean=clean, wrong=wrong)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/check", methods=["POST"])
@login_required
def api_check():
    """Step 1: rule-based logic check (correct / wrong + corrected code)."""
    code = request.json.get("code", "")
    if not code.strip():
        return jsonify({"error": "Please paste some Python code first."}), 400
    report = analyze_logic(code)
    return jsonify({
        "verdict": "correct" if report["ok"] else "wrong",
        "issues": report["issues"],
        "corrected_code": report.get("corrected_code"),
        "syntax_error": report.get("syntax_error", False),
        "no_while": report.get("no_while", False),
        "states": report.get("states", []),
        "regions": report.get("regions"),
    })


@app.route("/api/analyze_ml", methods=["POST"])
@login_required
def api_analyze_ml():
    """Step 2: ML AST/Gries analysis of the (corrected) code + save history."""
    code = request.json.get("code", "")
    source_verdict = request.json.get("verdict", "unknown")
    if not code.strip():
        return jsonify({"error": "No code to analyze."}), 400

    # AST features & tree
    feats, reg = extract_features(code)
    tree_text = ast_tree_text(code)

    # ML prediction (Gries 7-class)
    pred = ml_engine.predict_code(code)

    # per-code charts
    from engine.ml_engine import chart_state_features, chart_prediction_proba
    chart_feats = chart_state_features(feats)
    chart_proba = chart_prediction_proba(pred["ranking"])

    regions = {}
    if reg.get("while") is not None:
        regions = {"S": reg["S_lines"], "T": reg["T_lines"], "E": reg["E_lines"]}

    # save history
    hist = _load(HIST_FILE, [])
    rec = {
        "uid": session["uid"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "code": code,
        "verdict": source_verdict,
        "ml_label": pred["prediction"],
        "confidence": round(pred["ranking"][0]["prob"] * 100, 1),
        "id": hashlib.md5((code + str(datetime.now())).encode()).hexdigest()[:8],
    }
    hist.append(rec)
    _save(HIST_FILE, hist)

    return jsonify({
        "features": feats,
        "feature_names": FEATURE_NAMES,
        "tree_text": tree_text,
        "prediction": pred["prediction"],
        "prediction_code": pred["prediction_code"],
        "ranking": pred["ranking"],
        "regions": regions,
        "chart_features": chart_feats,
        "chart_proba": chart_proba,
    })


@app.route("/api/history")
@login_required
def api_history():
    hist = [h for h in _load(HIST_FILE, []) if h["uid"] == session["uid"]]
    hist.reverse()
    return jsonify(hist[:50])


@app.route("/api/dataset")
@login_required
def api_dataset():
    df = pd.read_csv(DATA_CSV)
    page = int(request.args.get("page", 0))
    size = int(request.args.get("size", 15))
    q = request.args.get("q", "").lower()
    label = request.args.get("label", "")
    if q:
        df = df[df["Description"].str.lower().str.contains(q) |
                df["Solution"].str.lower().str.contains(q)]
    if label:
        df = df[df["Label"] == label]
    total = len(df)
    batch = df.iloc[page * size: (page + 1) * size]
    return jsonify({
        "total": total,
        "rows": batch[["Description", "Solution", "Feedback", "Label", "Label_Code"]]
        .to_dict(orient="records"),
        "labels": sorted(pd.read_csv(DATA_CSV)["Label"].unique().tolist()),
    })


@app.route("/api/dataset/download")
@login_required
def dataset_download():
    return send_file(DATA_CSV, as_attachment=True,
                     download_name="gries_loop_tasks_codelens.csv")


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    bundle = ml_engine.get_bundle()
    hist = [h for h in _load(HIST_FILE, []) if h["uid"] == session["uid"]]
    charts = {
        "models": ml_engine.chart_model_comparison(bundle),
        "distribution": ml_engine.chart_label_distribution(bundle),
        "confusion": ml_engine.chart_confusion_matrix(bundle),
        "importance": ml_engine.chart_feature_importance(bundle),
    }
    metrics = bundle["metrics"].to_dict(orient="records")
    total = len(bundle["df"])
    user_total = len(hist)
    user_wrong = sum(1 for h in hist if h["verdict"] == "wrong")
    return jsonify({"charts": charts, "metrics": metrics,
                    "dataset_size": total, "user_total": user_total,
                    "user_wrong": user_wrong})


if __name__ == "__main__":
    # ensure the trained model bundle exists
    ml_engine.get_bundle()
    app.run(host="0.0.0.0", port=5000, debug=False)
