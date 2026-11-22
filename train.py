
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "epl_final.csv"
WINDOW = 5                                  # form = average of last 5 matches
STATS = ["GF", "GA", "Pts"]                 # goals for, goals against, points
FORM = [f"{s}_form" for s in STATS]
FEATURES = [f"{side}_{c}" for side in "HA" for c in FORM]


# 1. DATA -------------------------------------------------------------------
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"MatchDate": "Date", "FullTimeHomeGoals": "FTHG",
                            "FullTimeAwayGoals": "FTAG", "FullTimeResult": "FTR"})
    df = df[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"loaded {len(df)} matches, seasons {df.Season.iloc[0]} to {df.Season.iloc[-1]}")
    return df


# 2. FEATURES ---------------------------------------------------------------
def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with rolling form.

    *_form  -> uses only PREVIOUS matches (shift(1)); safe for training.
    *_now   -> includes the latest match; used to predict future fixtures.
    """
    home = pd.DataFrame({"Date": df.Date, "Team": df.HomeTeam, "GF": df.FTHG,
                         "GA": df.FTAG, "Pts": df.FTR.map({"H": 3, "D": 1, "A": 0})})
    away = pd.DataFrame({"Date": df.Date, "Team": df.AwayTeam, "GF": df.FTAG,
                         "GA": df.FTHG, "Pts": df.FTR.map({"A": 3, "D": 1, "H": 0})})
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Team", "Date"]).reset_index(drop=True)

    grouped = long.groupby("Team")
    for s in STATS:
        roll = lambda x: x.rolling(WINDOW, min_periods=3).mean()
        long[f"{s}_form"] = grouped[s].transform(lambda x: roll(x.shift(1)))  # no leakage
        long[f"{s}_now"] = grouped[s].transform(roll)
    return long


def build_dataset(df: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    def side(prefix, team_col):
        renames = {"Team": team_col, **{c: f"{prefix}_{c}" for c in FORM}}
        return long[["Date", "Team"] + FORM].rename(columns=renames)

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
    print(f"{len(data)} matches with usable features")

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

    # Latest form for every team in the most recent season of the data
    last = df[df.Season == df.Season.iloc[-1]]
    current = set(last.HomeTeam) | set(last.AwayTeam)
    latest = long.groupby("Team").tail(1).set_index("Team")
    form = {t: {s: round(float(latest.loc[t, f"{s}_now"]), 3) for s in STATS}
            for t in sorted(current) if t in latest.index and latest.loc[t, "GF_now"] == latest.loc[t, "GF_now"]}

    joblib.dump({"model": final, "features": FEATURES, "form": form, "accuracy": float(acc)},
                "model.joblib")
    print(f"\nSaved model.joblib with {len(form)} teams")


if __name__ == "__main__":
    main()