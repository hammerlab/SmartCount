import logging
logger = logging.getLogger(__name__)


def filter(df, mask, name=None, log=True):
    n_bef = len(df)
    df = df.loc[mask]
    if log:
        n_aft = len(df)
        diff = n_bef - n_aft
        pct = 100 * diff / n_bef if n_bef > 0 else 100
        name = '[Filter = {}] '.format(name) if name else ''
        logger.info('{}Removing {} records of {} ({:.2f}%)'.format(name, diff, n_bef, pct))
    return df
