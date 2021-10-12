from pathlib import Path
import requests

DATA_PATH = Path('flower_photos')
DATA_PATH.mkdir(parents=False, exist_ok=True)
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/"
FILENAME = "flower_photos.tgz"

if not (DATA_PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (DATA_PATH / FILENAME).open("wb").write(content)
