# Experiments

This document describes the steps took in the preparation of the data, training
and final selection of the model this repository holds.

## Dataset technical descriptions

[The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
presents with more than 7,000 images divided across 37 different breeds
of the domestic cats and dogs existent today.

| ![Oxford-IIIT Pet Dataset Statistics](https://www.robots.ox.ac.uk/~vgg/data/pets/breed_count.jpg) |
| :-: |
| *Dataset Statistics.* |

The dataset is available with the following directory structure:

| ![Oxford-IIIT Pet Dataset Directory Structure](oxford-iiit-dir-structure.png) |
| :-: |
| *Depiction of the Oxford-IIIT Pet Dataset directory structure.* |

where the `annotations.tar.gz` contains various metadata information regarding
the cats and dogs analyzed in their research paper, including a text file called
`list.txt` in which all the images labels are listed in a tabular format.
Meanwhile the `images.tar.gz` file contains the corpora of images that will be
the focus from here forward.

The `images.tar.gz` folder looks like this:

| ![Oxford-IIIT Pet Images Directory Structure](oxford-iiit-images-dir-structure.png) |
| :-: |
| *Depiction of the Oxford-IIIT Pet Images directory structure.* |

Each file inside the `images.tar.gz` belongs to a sample of each breed
described earlier, with the exception of three `.mat` files that appears to be
there human error and will be ignored.

Notice how the images are labeled. They represent the images that belongs to
cats using the uppercase letter of the breed itself and the lowercase letter
for dogs. After the breed name in this format an autoincrement identifier is
appended.
