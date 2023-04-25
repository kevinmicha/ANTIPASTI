import os

TEST_PATH = os.path.dirname(__file__)
OUT_PATH = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)
