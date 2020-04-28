from catboost import CatBoostClassifier
from common import get_train_data_and_labels


def main():
    print('-- Start --')

    labels = ['barash', 'car-carich', 'copatich', 'crosh', 'ejik', 'losyash', 'nusha', 'pin', 'sovunja']
    train_data, train_labels = get_train_data_and_labels('../images/balls-for-study', labels)

    model = CatBoostClassifier(
        iterations=10,
        # task_type='GPU',
    )
    model.fit(train_data, train_labels)  # обучение классификатора

    model.save_model('model')

    print('-- End --')


if __name__ == '__main__':
    main()
