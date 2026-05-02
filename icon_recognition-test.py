from PIL import Image
import imagehash
import os

test_cases = os.listdir('icon_recognition_test_cases')

for test_case in test_cases:
    files = os.listdir(f'icon_recognition_test_cases/{test_case}')
    input_hashes = {}
    template_hash = ''

    for file in files:
        image = Image.open(f'icon_recognition_test_cases/{test_case}/{file}')

        if file.startswith('template'):
            template_hash = imagehash.phash(image)
        else:
            input_hashes[file] = imagehash.phash(image)

    print(f'\nTest case: {test_case}')
    for file, input_hash in input_hashes.items():
        distance = input_hash - template_hash
        print(f'\t{file} perceptual hash error: {distance / 64:.4f}')
