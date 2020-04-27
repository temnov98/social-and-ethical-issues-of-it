from common import handle_images_for_study


def main():
    dimension = (200, 200)  # (width, height)

    balls = ['barash', 'car-carich', 'copatich', 'crosh', 'ejik', 'losyash', 'nusha', 'pin', 'sovunja']

    for ball in balls:
        print(f'Started {ball}')
        handle_images_for_study(dimension, f'../images/raw-balls/{ball}', f'../images/balls/{ball}')

        print()


if __name__ == '__main__':
    main()
