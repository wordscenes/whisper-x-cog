#!/usr/bin/env python

import sys

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import Predictor

# Running prediction once will trigger the download of the model
p = Predictor()
p.setup()
