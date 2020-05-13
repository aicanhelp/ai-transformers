from transformersx.models import *


def test_models():
    for model in Tiny().models():
        print(model.path)


def test_model_sizes():
    for m in models("tiny,base"):
        print(m.path)
