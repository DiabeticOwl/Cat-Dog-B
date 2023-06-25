# Cat-Dog-B

Model that attempts to predict whether a passed image belongs to a Cat or a Dog
and it's breed.

## Data used

The main data used for training the model comes from the
[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), a set
of 37 categories of cat and dog breeds.

Their [published paper](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)
describes the study and creation of a model which can discriminate the collected
37 classes of breed. Their approach is categorized by the use of multiple layers
from training with the entire image towards segmenting each one and focusing the
model on specific parts of the detected animal, like the head formation. Another
challenge that they beat was the manual collection of the dataset as they
described two main issues in doing so in the paper:

> The difficulty is in the fact that breeds may differ only by a few subtle
> phenotypic details that, due to the highly deformable nature of the bodies of
> such animals, can be difficult to measure automatically.

and:

> It is not unusual for owners to believe (and post) the incorrect breed for
> their pet

Alongside their contribution of 37 breeds 2 more were included to the model
presented in this repository with the intent of proving the versatility of the
selected model class and also how far deep learning models in this specific
problem of differentiate animal breeds based solely in their images have become.

## Experiments

The final model was chosen from a set of multiple experiments which are
described [here](https://github.com/DiabeticOwl/Cat-Dog-B/blob/main/experiments/README.md).

## Demo

A demo app was created using [Gradio](https://gradio.app/) and hosted on
[Hugging Face](https://huggingface.co/) for public use.

For trying this model please refer to [this]() url.
