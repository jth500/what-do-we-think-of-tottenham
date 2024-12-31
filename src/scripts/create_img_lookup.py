import pandas as pd

from wdwtot.constants import ROOT

if __name__ == "__main__":
    d = ROOT / "data/processed"
    files = list(d.glob("**/*.png"))

    df = pd.DataFrame(files, columns=["path"])
    df["team_name"] = (
        df["path"].apply(lambda x: x.stem).str.split("_").apply(lambda x: x[0])
    )
    df = df.reset_index()
    df = df[["path", "index", "team_name"]]

    # from string label, create a numeric label
    df["label"] = pd.Categorical(df["team_name"])
    df["label"] = df["label"].cat.codes
    df.sort_values("label", inplace=True)
    print(df.columns)

    df.to_csv(ROOT / "data/processed/logo-lookup.csv", index=False)
