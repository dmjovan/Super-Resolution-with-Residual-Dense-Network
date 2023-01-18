import logging


def get_logger(module_name: str) -> logging.Logger:

    """ Setting up logger for project. """

    logging.basicConfig(filename="log.txt")

    # Creating logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Creating console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Creating formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Adding formatter to ch
    ch.setFormatter(formatter)

    # Adding ch to logger
    logger.addHandler(ch)

    return logger
