import os
import glob
import pandas as pd

RAW_FOLDER = "raw_data"
OUT_FILE = "processed_data/mixed_dataset.csv"
RANDOM_SEED = 42  # change or set None for different shuffle each run

# map each file to its gesture
map_gesture = {
    "round1.csv":"round",
    "round3.csv":"round",
    "shoot3.csv":"shoot",
    "shoot_gesture1.csv":"shoot",
    "up_down1.csv":"up_down",
    "up_down3.csv":"up_down",
}

def main():
    csv_paths = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {RAW_FOLDER}/")

    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"Skipping empty file: {path}")
                continue

            # Optional: keep track of where each row came from
            df["gesture"] = map_gesture[str(os.path.basename(path))]

            dfs.append(df)
            print(f"Loaded {path} -> {len(df)} rows")

        except Exception as e:
            print(f"Skipping bad file {path}: {e}")

    if not dfs:
        raise ValueError("All CSV files were empty or unreadable.")

    # Concatenate (union of columns; missing values become NaN)
    big_df = pd.concat(dfs, ignore_index=True, sort=False)

    print(f"\nTotal rows before shuffle: {len(big_df)}")

    # Shuffle randomly
    big_df = big_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Saving mixed dataset -> {OUT_FILE}")
    big_df.to_csv(OUT_FILE, index=False)
    print("Done")

if __name__ == "__main__":
    main()
