import csv
import json

settings_file = "settings.json"

def load_settings():
    with open(settings_file) as f:
        settings = json.load(f )

    return settings


def save_settings(data):
    with open(settings_file, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4,
                  ensure_ascii=False)

settings = load_settings()
print("Model",settings["model"] )
print("Model File",settings["model_file"] )

settings["model_file"] = "test"

save_settings(settings)

