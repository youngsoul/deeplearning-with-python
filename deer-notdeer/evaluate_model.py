from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.utils import image_dataset_from_directory
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb

"""
Usage:

python evaluate_model.py --dataset ./datasets/test

python evaluate_model.py

python evaluate_model.py --dataset /Users/patrickryan/Development/mygithub/deeplearning-with-python/datasets/backyard/images
"""

def _load_test_dataset(dataset_dir):

    test_dataset = image_dataset_from_directory(
        Path(dataset_dir),
        image_size=(224,224),
        batch_size=32
    )

    # Page 219 Deep Learning with Python 2nd Edition
    # Use Rescaling layer in the DataSet API
    # To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.
    normalization_layer = Rescaling(scale= (1./127.5), offset=-1)
    normalized_test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return normalized_test_ds

def main():
    model = load_model('./model_checkpoints')
    print(model.summary())

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, default="./datasets/test",
                    help="path to input dataset")

    args = vars(ap.parse_args())

    test_dataset = _load_test_dataset(args['dataset'])

    evaluate_model_on_dataset(model, test_dataset)


def evaluate_model_on_dataset(model, test_dataset):
    # Evaluate on the test dataset as specified from cmd line options.
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")

    # for each image in test dataset, predict for each image.
    label_names = ['Background', 'Deer']
    for data_batch in test_dataset:
        # data_batch is a batch of 32 images
        batch_of_images = data_batch[0]
        batch_of_labels = data_batch[1]

        batch_of_preds = model.predict(batch_of_images)

        for image, label, pred in zip(batch_of_images, batch_of_labels, batch_of_preds):
            label = kb.get_value(label)
            pred = kb.get_value(pred)[0]
            if pred >= 0.5:
                print(f"Label: {label_names[label]}, Pred: Deer")
            else:
                print(f"Label: {label_names[label]}, Pred: Background")

            # rescale the image from the dataset to be an actual image

            if label != int(round(pred,0)):
                x = image.numpy() + 1
                x = x * 127.5
                x = x.astype(int)
                plt.imshow(x)
                plt.show()


if __name__ == '__main__':
    main()
