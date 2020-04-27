from common import handle_images_for_study


def main():
    dimension = (200, 200)  # (width, height)

    # можно было бы использовать handle_images_for_tests,
    handle_images_for_study(dimension, f'./images/raw-balls', f'./images/balls')


if __name__ == '__main__':
    main()
