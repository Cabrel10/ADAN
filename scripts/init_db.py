import optuna
import os

def main():
    storage = "sqlite:///optuna.db"
    workers = ["w1", "w2", "w3", "w4"]
    
    for w in workers:
        study_name = f"adan_final_v1_{w}"
        print(f"Creating/Loading study {study_name}...")
        optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
    print("DB Initialized.")

if __name__ == "__main__":
    main()
