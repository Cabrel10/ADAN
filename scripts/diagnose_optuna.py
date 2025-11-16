import optuna
import json

def diagnose_trial(trial_number: int):
    """Load a specific trial and print its user_attrs."""
    try:
        study = optuna.load_study(
            study_name="adan_final_v1", storage="sqlite:///optuna.db"
        )
        trial = study.get_trials()[trial_number]
        print(f"--- User Attributes for Trial {trial.number} ---")
        print(json.dumps(trial.user_attrs, indent=2, default=str))
    except IndexError:
        print(f"Error: Trial {trial_number} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # We know from the logs that trial 3 had trades.
    diagnose_trial(3)
