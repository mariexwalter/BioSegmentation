import subprocess
import shutil
import re
from glob import glob
from os.path import join, basename
from os import makedirs
import os
BASE_FOLDER = join('data', 'breast_cancer')


def run():
    os.chdir('data')
    subprocess.call(['unzip', 'archive.zip'])
    os.chdir('..')

    def is_test(x):
        x = basename(x)
        return bool(re.match(r'ytma55_(.*)(.xml|.tif|.TIF)', x))

    for set_folder in ['train', 'test']:
        for type_folder in ['images', 'labels']:
            makedirs(
                join(BASE_FOLDER, set_folder, type_folder),
                exist_ok=True
            )

    im_folder = 'data/Images'
    for im_path in glob(join(im_folder, '*')):
        folder = 'test' if is_test(im_path) else 'train'
        # image: xxx_ccd_.tif, label: xxx.TIF
        new_im_name = basename(im_path).replace('_ccd', '')
        shutil.move(
            im_path,
            join(BASE_FOLDER, folder, 'images', new_im_name)
        )

    label_folder = 'data/Masks'
    for mask_path in glob(join(label_folder, '*')):
        folder = 'test' if is_test(mask_path) else 'train'
        shutil.move(
            mask_path,
            join(BASE_FOLDER, folder, 'labels', basename(mask_path))
        )


if __name__ == '__main__':
    run()
