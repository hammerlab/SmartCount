import pandas as pd


def group_dates(dates, min_gap_seconds):
    # Make sure dates are sorted
    if not dates.is_monotonic_increasing:
        dates = dates.sort_values()

    # Make sure dates are also unique
    if not dates.is_unique:
        dates = dates.drop_duplicates()

    # Create a new group index each time the difference between steps exceeds the given threshold (in seconds)
    groups = (dates.diff().dt.seconds >= min_gap_seconds).cumsum()

    # Get the minimum date for each group and then get a vector of len(dates) continaing the group date
    # for each original date
    groups = groups.map(dates.groupby(groups).min())

    # Return a series mapping the original dates to the grouped date
    return pd.Series(groups.values, index=dates.values)


