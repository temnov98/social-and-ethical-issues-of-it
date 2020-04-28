from catboost import CatBoostClassifier
from common import read_train_image
import pathlib


def main():
    model = CatBoostClassifier()
    # model.load_model('model_barash_ejik_carich')
    model.load_model('model')

    data_dir = pathlib.Path('../images/balls-for-predict')

    for path in data_dir.glob('*'):
        print(path)
        image = read_train_image(path)
        prediction = model.predict([image])
        print(prediction)

    print('End')


if __name__ == '__main__':
    main()
