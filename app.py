
import math
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
bundle = joblib.load("model.joblib")          # created by train.py
model, FEATURES, FORM = bundle["model"], bundle["features"], bundle["form"]

# The project data and saved model can lag behind the current league season.
# Keep the choices aligned with the 2026/27 Premier League lineup, while using
# each club's latest available form whenever it is present in the model bundle.
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Coventry", "Crystal Palace", "Everton", "Fulham", "Hull",
    "Ipswich", "Leeds", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nott'm Forest", "Sunderland", "Tottenham",
]
STATS = ("GF", "GA", "Pts")
TABLE_STATS = ("Pts_table", "GD_table")
DATA_AS_OF = bundle.get("data_as_of")
TABLE_AS_OF = "2026-11-22"
# Snapshot used only while the saved model predates the current campaign. Refresh
# the model with train2.py to replace this fallback with match-by-match features.
CURRENT_TABLE = {
    "Arsenal": (28, 10, 11), "Brighton": (22, 17, 11), "Man City": (22, 8, 11),
    "Bournemouth": (18, 4, 11), "Liverpool": (18, 4, 10), "Leeds": (17, 5, 11),
    "Everton": (16, 2, 10), "Chelsea": (16, 0, 11), "Tottenham": (15, 1, 11),
    "Brentford": (14, 5, 10), "Hull": (14, 2, 11), "Aston Villa": (14, -3, 11),
    "Newcastle": (13, -3, 11), "Coventry": (12, -8, 11), "Fulham": (11, -2, 11),
    "Ipswich": (11, -7, 11), "Nott'm Forest": (9, -6, 11), "Sunderland": (9, -9, 11),
    "Man United": (8, -5, 10), "Crystal Palace": (7, -15, 11),
}


def apply_current_table_context(home, away, probabilities):
    """Nudge an older model using this season's table and home advantage."""
    if DATA_AS_OF and DATA_AS_OF >= TABLE_AS_OF:
        return probabilities
    if home not in CURRENT_TABLE or away not in CURRENT_TABLE:
        return probabilities

    def rating(team):
        points, goal_difference, matches = CURRENT_TABLE[team]
        return 0.65 * points / matches + 0.25 * goal_difference / matches

    # Apply a bounded, symmetric adjustment so table strength affects the
    # home/away balance while preserving the model's draw estimate as context.
    shift = max(-1.5, min(1.5, rating(home) - rating(away) + 0.25))
    factor = math.exp(shift / 2)
    weighted = {
        "home": probabilities["home"] * factor,
        "draw": probabilities["draw"],
        "away": probabilities["away"] / factor,
    }
    total = sum(weighted.values())
    return {key: value / total for key, value in weighted.items()}


def add_historical_form_for_missing_teams():
    """Fill missing current-club form from the latest five local match records."""
    missing = set(CURRENT_TEAMS) - set(FORM)
    if not missing:
        return

    try:
        matches = pd.read_csv("epl_final.csv", parse_dates=["MatchDate"])
    except (OSError, ValueError, pd.errors.ParserError):
        return

    for team in missing:
        home = matches.loc[matches["HomeTeam"] == team]
        away = matches.loc[matches["AwayTeam"] == team]
        home = pd.DataFrame({
            "Date": home["MatchDate"], "Season": home["Season"] if "Season" in home else None,
            "GF": home["FullTimeHomeGoals"],
            "GA": home["FullTimeAwayGoals"],
            "Pts": home["FullTimeResult"].map({"H": 3, "D": 1, "A": 0}),
        })
        away = pd.DataFrame({
            "Date": away["MatchDate"], "Season": away["Season"] if "Season" in away else None,
            "GF": away["FullTimeAwayGoals"],
            "GA": away["FullTimeHomeGoals"],
            "Pts": away["FullTimeResult"].map({"A": 3, "D": 1, "H": 0}),
        })
        played = pd.concat([home, away], ignore_index=True).sort_values("Date").tail(5)
        if played.empty:
            continue
        all_team_matches = pd.concat([home, away], ignore_index=True)
        source_season = matches.loc[
            (matches["HomeTeam"] == team) | (matches["AwayTeam"] == team), "Season"
        ].max()
        season_matches = (all_team_matches.loc[all_team_matches["Season"] == source_season]
                          if "Season" in matches else all_team_matches)
        FORM[team] = {
            "GF": round(float(played["GF"].mean()), 3),
            "GA": round(float(played["GA"].mean()), 3),
            "Pts": round(float(played["Pts"].mean()), 3),
            "Pts_table": round(float(season_matches["Pts"].mean()), 3),
            "GD_table": round(float((season_matches["GF"] - season_matches["GA"]).mean()), 3),
        }


add_historical_form_for_missing_teams()

if not DATA_AS_OF:
    try:
        DATA_AS_OF = pd.to_datetime(pd.read_csv("epl_final.csv", usecols=["MatchDate"])["MatchDate"]).max().strftime("%Y-%m-%d")
    except (OSError, ValueError, pd.errors.ParserError):
        DATA_AS_OF = None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/teams")
def teams():
    return jsonify(teams=[team for team in CURRENT_TEAMS if team in FORM], accuracy=bundle["accuracy"],
                   data_as_of=DATA_AS_OF, table_as_of=TABLE_AS_OF)


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    home, away = body.get("home"), body.get("away")
    if home not in CURRENT_TEAMS or away not in CURRENT_TEAMS or home not in FORM or away not in FORM:
        return jsonify(error="Choose two teams from the lists."), 400
    if home == away:
        return jsonify(error="Pick two different teams."), 400

    row = {}
    for feature in FEATURES:
        side, stat = feature.split("_", 1)
        team = home if side == "H" else away
        key = stat.removesuffix("_form")
        row[feature] = FORM[team].get(key, 0.0)
    X = pd.DataFrame([row])[FEATURES]

    probs = dict(zip(model.classes_, model.predict_proba(X)[0]))   # keys: A, D, H
    outcome_probs = apply_current_table_context(home, away, {
        "home": probs.get("H", 0), "draw": probs.get("D", 0), "away": probs.get("A", 0),
    })
    return jsonify(
        home=home, away=away,
        probs=outcome_probs,
        form={home: FORM[home], away: FORM[away]},
    )


if __name__ == "__main__":
    app.run(debug=True)
