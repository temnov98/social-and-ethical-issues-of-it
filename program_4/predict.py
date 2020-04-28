from catboost import CatBoostClassifier
from common import read_train_image


def main():
    model = CatBoostClassifier()
    # model.load_model('model_barash_ejik_carich')
    model.load_model('model-10')

    test_data = [
        read_train_image('./images/balls-for-predict/0_0.png'),  # pin
        read_train_image('./images/balls-for-predict/0_1.png'),  # pin
        read_train_image('./images/balls-for-predict/0_2.png'),  # pin

        read_train_image('./images/balls-for-predict/1_0.png'),  # nusha
        read_train_image('./images/balls-for-predict/1_1.png'),  # nusha
        read_train_image('./images/balls-for-predict/1_2.png'),  # nusha

        read_train_image('./images/balls-for-predict/2_0.png'),  # losyash
        read_train_image('./images/balls-for-predict/2_1.png'),  # losyash
        read_train_image('./images/balls-for-predict/2_2.png'),  # losyash

        read_train_image('./images/balls-for-predict/3_0.png'),  # sovunia
        read_train_image('./images/balls-for-predict/3_1.png'),  # sovunia
        read_train_image('./images/balls-for-predict/3_2.png'),  # sovunia

        read_train_image('./images/balls-for-predict/4_0.png'),  # krosh
        read_train_image('./images/balls-for-predict/4_1.png'),  # krosh
        read_train_image('./images/balls-for-predict/4_2.png'),  # krosh

        read_train_image('./images/balls-for-predict/5_0.png'),  # kopatich
        read_train_image('./images/balls-for-predict/5_1.png'),  # kopatich
        read_train_image('./images/balls-for-predict/5_2.png'),  # kopatich

        read_train_image('./images/balls-for-predict/6_0.png'),  # kar-karich
        read_train_image('./images/balls-for-predict/6_1.png'),  # kar-karich
        read_train_image('./images/balls-for-predict/6_2.png'),  # kar-karich

        read_train_image('./images/balls-for-predict/7_0.png'),  # ejik
        read_train_image('./images/balls-for-predict/7_1.png'),  # ejik
        read_train_image('./images/balls-for-predict/7_2.png'),  # ejik

        read_train_image('./images/balls-for-predict/8_0.png'),  # barahs
        read_train_image('./images/balls-for-predict/8_1.png'),  # barahs
        read_train_image('./images/balls-for-predict/8_2.png'),  # barahs
    ]

    prediction = model.predict(test_data)

    print(prediction)
    print('End')


if __name__ == '__main__':
    main()
