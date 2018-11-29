from cvutils.visualize import display_images


def display_apartment_digit_images(apt_data, limit=5, cols=4, random_state=None):
    df = apt_data
    if limit is not None:
        df = df.sample(n=min(limit, len(apt_data)), random_state=random_state)
    num_images = [
        (
            r['apt_num_digit_images'] + r['st_num_digit_images'],
            r['apt_num'] + r['st_num']
        )
        for _, r in df.iterrows()
    ]
    display_images(
        [img for e in num_images for img in e[0]],
        titles=[digit for e in num_images for digit in e[1]],
        cols=cols
    )
