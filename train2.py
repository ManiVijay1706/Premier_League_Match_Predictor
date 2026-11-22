"""Download Premier League results, build pre-match form features,
train a random forest, and save everything the website needs.

Run:  python train2.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

WINDOW = 5                                  # form = average of last 5 matches
STATS = ["GF", "GA", "Pts"]                 # goals for, goals against, points
FORM = [f"{s}_form" for s in STATS]
TABLE_STATS = ["Pts_table", "GD_table"]      # season-to-date points and goal difference per match
TABLE_NOW = [f"{s}_now" for s in TABLE_STATS]
FEATURES = [f"{side}_{c}" for side in "HA" for c in FORM + TABLE_STATS]
FIRST_SEASON, LAST_SEASON = 2014, 2026      # season start years


# 1. DATA -------------------------------------------------------------------
def load_matches() -> pd.DataFrame:
    frames = []
    for y in range(FIRST_SEASON, LAST_SEASON + 1):
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"          # e.g. 2425
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        try:
            frames.append(pd.read_csv(url, encoding="latin-1",
                          storage_options={"User-Agent": "Mozilla/5.0"}))
            print(f"loaded season {code}")
        except Exception as err:
            print(f"skipped {code}: {err}")

    if not frames:  # offline fallback: put E0.csv files in ./data
        frames = [pd.read_csv(p, encoding="latin-1") for p in sorted(Path("data").glob("*.csv"))]
    if not frames:
        raise SystemExit("No data. Check your internet or put CSVs in ./data")

    df = pd.concat(frames, ignore_index=True)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    # Season starts in July; this also handles datasets without a Season column.
    df["Season"] = df["Date"].dt.year - (df["Date"].dt.month < 7).astype(int)
    return df.sort_values("Date").reset_index(drop=True)


# 2. FEATURES ---------------------------------------------------------------
def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with rolling form.

    *_form  -> uses only PREVIOUS matches (shift(1)); safe for training.
    *_now   -> includes the latest match; used to predict future fixtures.
    """
    home = pd.DataFrame({"Date": df.Date, "Season": df.Season, "Team": df.HomeTeam, "GF": df.FTHG,
                         "GA": df.FTAG, "Pts": df.FTR.map({"H": 3, "D": 1, "A": 0})})
    away = pd.DataFrame({"Date": df.Date, "Season": df.Season, "Team": df.AwayTeam, "GF": df.FTAG,
                         "GA": df.FTHG, "Pts": df.FTR.map({"A": 3, "D": 1, "H": 0})})
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Team", "Season", "Date"]).reset_index(drop=True)

    grouped = long.groupby("Team", sort=False)
    for s in STATS:
        roll = lambda x: x.rolling(WINDOW, min_periods=3).mean()
        long[f"{s}_form"] = grouped[s].transform(lambda x: roll(x.shift(1)))  # no leakage
        long[f"{s}_now"] = grouped[s].transform(roll)

    long["GD"] = long["GF"] - long["GA"]
    season = long.groupby(["Team", "Season"], sort=False)
    long["Pts_table"] = season["Pts"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(1.0)
    long["GD_table"] = season["GD"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0.0)
    long["Pts_table_now"] = season["Pts"].transform(lambda x: x.expanding().mean())
    long["GD_table_now"] = season["GD"].transform(lambda x: x.expanding().mean())
    return long


def build_dataset(df: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    def side(prefix, team_col):
        renames = {"Team": team_col, **{c: f"{prefix}_{c}" for c in FORM}}
        return long[["Date", "Team"] + FORM + TABLE_STATS].rename(columns=renames)

    data = (df.merge(side("H", "HomeTeam"), on=["Date", "HomeTeam"])
              .merge(side("A", "AwayTeam"), on=["Date", "AwayTeam"])
              .dropna()
              .sort_values("Date")
              .reset_index(drop=True))
    return data


# 3. TRAIN + EVALUATE -------------------------------------------------------
def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=6,
                                  min_samples_leaf=10, random_state=42, n_jobs=-1)


def main():
    df = load_matches()
    long = build_long(df)
    data = build_dataset(df, long)
    print(f"\n{len(data)} matches with usable features")

    X, y = data[FEATURES], data["FTR"]

    # Time-based split: train on the past, test on the most recent 20%
    split = int(len(data) * 0.8)
    model = make_model().fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    acc = accuracy_score(y.iloc[split:], pred)
    baseline = (y.iloc[split:] == "H").mean()
    print(f"Test accuracy: {acc:.3f}   (always-pick-home baseline: {baseline:.3f})")
    print(classification_report(y.iloc[split:], pred, zero_division=0))

    print("Feature importance:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:12s} {imp:.3f}")

    # Refit on ALL matches for the model the website will use
    final = make_model().fit(X, y)

    # Latest form for every current team (teams seen in the last ~380 matches)
    current = set(df.tail(380).HomeTeam) | set(df.tail(380).AwayTeam)
    latest = long.groupby("Team").tail(1).set_index("Team")
    form = {
        t: ({s: round(float(latest.loc[t, f"{s}_now"]), 3) for s in STATS}
            | {s: round(float(latest.loc[t, f"{s}_now"]), 3) for s in TABLE_STATS})
            for t in sorted(current) if t in latest.index and latest.loc[t, "GF_now"] == latest.loc[t, "GF_now"]}

    joblib.dump({"model": final, "features": FEATURES, "form": form, "accuracy": float(acc),
                 "data_as_of": df.Date.max().strftime("%Y-%m-%d")},
                "model.joblib")
    print(f"\nSaved model.joblib with {len(form)} teams")


if __name__ == "__main__":
    main()
