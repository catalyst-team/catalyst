import time
from IPython.display import display, Javascript
import hashlib


def save_notebook(filepath: str, wait_period: float = 0.1, max_wait_time=1.0):
    start_md5 = hashlib.md5(open(filepath, "rb").read()).hexdigest()
    display(Javascript("IPython.notebook.save_checkpoint();"))
    current_md5 = start_md5

    wait_time = 0
    while start_md5 == current_md5:
        time.sleep(wait_period)
        wait_time += wait_period
        current_md5 = hashlib.md5(open(filepath, "rb").read()).hexdigest()
        if wait_time > max_wait_time:
            break
