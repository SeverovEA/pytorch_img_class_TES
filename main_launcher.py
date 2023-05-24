import tkinter as tk
from tkinter.filedialog import askopenfilename
import subprocess
import yaml
import os
import configparser
from pathlib import Path

CONFIG_FILENAME = "config.yaml"
SETTINGS_FILENAME = "settings.yaml"
image_path = ""

with open(SETTINGS_FILENAME, "r") as stream:
    settings_dict = yaml.safe_load(stream)
with open(CONFIG_FILENAME, "r") as stream:
    config_dict = yaml.safe_load(stream)


def main():
    global image_path
    root = tk.Tk()
    root.title("Launcher")
    model_path = tk.StringVar()
    model_path.set(str(Path(settings_dict["model_path"])))
    img_path = tk.StringVar()
    img_path.set(str(Path(settings_dict["img_path"])))
    image_label = tk.Label(root, text="Image path:")
    image_entry = tk.Entry(
        root,
        textvariable=img_path,
        width=100,
    )
    image_browse_button = tk.Button(
        root,
        text="Select different image",
        command=lambda: browse("image", img_path)
    )
    model_label = tk.Label(root, text="Model path:")
    model_entry = tk.Entry(
        root,
        textvariable=model_path,
        width=100,
    )
    model_browse_button = tk.Button(
        root,
        text="Select different model",
        command=lambda: browse("model", model_path),
    )
    save_button = tk.Button(
        root,
        text="Save Settings",
        command=lambda: save_settings(img_entry=image_entry, model_entry=model_entry)
    )
    run_button = tk.Button(
        root,
        text="Run Prediction",
        command=lambda: run_prediction()
    )

    image_label.grid(row=1, column=0, padx=10, pady=10)
    image_entry.grid(row=1, column=1, columnspan=3, padx=10, pady=10, sticky="ew")
    image_browse_button.grid(row=1, column=5, padx=10, pady=10)

    model_label.grid(row=2, column=0, padx=10, pady=10)
    model_entry.grid(row=2, column=1, columnspan=3, padx=10, pady=10, sticky="ew")
    model_browse_button.grid(row=2, column=5, padx=10, pady=10)

    run_button.grid(row=3, column=1, padx=10, pady=10, ipadx=10, ipady=5)
    save_button.grid(row=3, column=2, padx=10, pady=10, ipadx=10, ipady=5)

    root.resizable(False, False)
    root.mainloop()

    return image_path


def set_parameter(parameter: str, entry: tk.Entry) -> None:
    value = entry.get()
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_FILENAME)
    cfg.set(section="Paths", option=parameter, value=value)
    with open(CONFIG_FILENAME, "wb") as configfile:
        cfg.write(configfile)


def browse(mode: str, variable: tk.StringVar):
    if mode == "model":
        variable.set(str(Path(askopenfilename(initialdir=os.path.dirname(settings_dict["model_path"])))))
    elif mode == "image":
        variable.set(str(Path(askopenfilename(initialdir=os.path.dirname(settings_dict["img_path"])))))


def save_settings(img_entry: tk.Entry, model_entry: tk.Entry, settings: dict = settings_dict):
    settings["img_path"] = img_entry.get()
    settings["model_path"] = model_entry.get()
    with open("settings.yaml", "w") as file:
        yaml.safe_dump(settings, file, sort_keys=False, default_flow_style=False)


def run_prediction():
    subprocess.call(["python", "main_predict.py"])


if __name__ == "__main__":
    main()
