from common import handle_images_for_predict, default_dimension


def main():
    handle_images_for_predict(default_dimension, f'../images/raw-balls-for-predict', f'../images/balls-for-predict')

    print('End')


if __name__ == '__main__':
    main()
