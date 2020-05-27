from datetime import datetime
import matplotlib.pyplot as plt


def write_results(path, entry_text):
    with open(path, "a") as text_file:
        current_datetime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_datetime}, {entry_text}\n", file=text_file)

