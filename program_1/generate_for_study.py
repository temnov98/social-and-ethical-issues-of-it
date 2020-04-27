from common import handle_images_for_study, default_dimension


def main():
    balls = ['barash', 'car-carich', 'copatich', 'crosh', 'ejik', 'losyash', 'nusha', 'pin', 'sovunja']

    for ball in balls:
        print(f'Started {ball}')

        source_directory = f'../images/raw-balls-for-study/{ball}'
        destination_directory = f'../images/balls-for-study/{ball}'

        handle_images_for_study(default_dimension, source_directory, destination_directory)

        print()

    print('End')


if __name__ == '__main__':
    main()
