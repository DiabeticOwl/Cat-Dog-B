"""Downloads and extract data for the model."""
import gdown
import random
import requests
import tarfile
import zipfile

from pathlib import Path
from shutil import copy, move, rmtree
from typing import Literal, Optional
from tqdm.auto import tqdm

def _direct_download(source: str,
                     destination_path: Optional[Path] = Path('.')) -> Path:
    print(f"Downloading {destination_path.name} data...")
    with open(destination_path, "wb") as f:
        request = requests.get(source)
        f.write(request.content)
    return destination_path

def _drive_download(source: str,
                    destination_path: Optional[Path] = Path('.')) -> Path:
    output = None
    if destination_path.suffix == '.zip':
        # gdown accepts either a None or a string.
        output = str(destination_path.resolve())
    
    fn = gdown.download(source, output, fuzzy=True)
    
    # A None output will download the file in the current directory.
    # fn will be set equal to the file's name.
    # The downloaded file will be then moved to the desired destination path.
    if not output:
        destination_path.mkdir(parents=True, exist_ok=True)
        move(fn, destination_path)
    
    return destination_path.joinpath(fn)

_TYPES = Literal['direct', 'google-drive']
def extract_data(source: str,
                 destination: Optional[Path] = Path('.'),
                 remove_zip: Optional[bool] = True,
                 source_type: Optional[_TYPES] = 'direct',
                 output_name: Optional[str] = '') -> Path:
    """Downloads the file given by source and extracts it on destination.
    
    The data extracted by this function currently supports both direct and
    Google Drive links pointing to .zip files.
    
    Args:
        source: String containing the link where the file is stored. It
            expects it to be a zip file.
            Example: https://github.com/.../file.zip
        destination: String or Path pointing to where the file should be
            downloaded.
        remove_zip: Bool that determines whether the downloaded zi file
            would be deleted.
        source_type: String that will determine whether the file in source
            is from direct or Google Drive. A source of 'direct' type must
            point to the specific file or it may not work.
        output_name: If source_type is 'google-drive' it will concat
            this value to the destination path. Otherwise is ignored.

    Returns:
        The Path of the extracted directory.
    """
    if source_type == 'direct':
        output_name = source.split('/')[-1]

    if destination.is_dir():
        print(f"{destination} directory exists.")
    else:
        print(f"Did not find {destination} directory, creating it...")
        destination.mkdir(parents=True, exist_ok=True)

    dwn_dict = {
        'direct': _direct_download,
        'google-drive': _drive_download
    }
    destination /= output_name
    zipfile_path = dwn_dict[source_type](source, destination)
    # In case the file has multiple suffixes like ".tar.gz".
    file_path = zipfile_path.name.split('.')[0]

    print(f"Decompressing {zipfile_path.name} data...") 
    if ''.join(zipfile_path.suffixes) == ".tar.gz":
        with tarfile.open(zipfile_path, "r:gz") as tar:
            tar.extractall(file_path)
    else:
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(file_path)
        
    if remove_zip:
        zipfile_path.unlink()

    return Path(file_path)

if __name__ == "__main__":
    zip_url = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz'
    imgs_path = extract_data(zip_url) / 'images'

    print("Extracting classes info...")
    class_names = {' '.join(p.name.split('_')[:-1])
                   for p in imgs_path.glob('*.jpg')}
    breeds = {br_name: [] for br_name in class_names}

    print("Creating modelling structure...")
    data_path = Path('./restrc-oxford-iiit-pet')
    data_path.mkdir(parents=True, exist_ok=True)
    train_path = data_path / 'train'
    train_path.mkdir(parents=True, exist_ok=True)
    eval_path = data_path / 'eval'
    eval_path.mkdir(parents=True, exist_ok=True)
    test_path = data_path / 'test'
    test_path.mkdir(parents=True, exist_ok=True)

    print("Distributing classes...")
    for p in imgs_path.glob('*.jpg'):
        br_name = ' '.join(p.name.split('_')[:-1])
        breeds[br_name].append(p)
    random.seed(777)
    for breed, imgs in tqdm(breeds.items()):
        tr_path = train_path / breed
        tr_path.mkdir(parents=True, exist_ok=True)
        e_path = eval_path / breed
        e_path.mkdir(parents=True, exist_ok=True)
        ts_path = test_path / breed
        ts_path.mkdir(parents=True, exist_ok=True)
        
        shuf_imgs = random.sample(imgs, k=len(imgs))
        train_imgs = shuf_imgs[:50]
        eval_imgs = shuf_imgs[50:100]
        test_imgs = shuf_imgs[100:]
        
        for img in train_imgs:
            path = tr_path / img.name
            copy(img, path)
        for img in eval_imgs:
            path = e_path / img.name
            copy(img, path)
        for img in test_imgs:
            path = ts_path / img.name
            copy(img, path)
    print("Cleaning...")
    rmtree(imgs_path.parent)
