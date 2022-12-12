import logging 
logging.basicConfig(level=logging.DEBUG)


def test_logging():
    for i in range(100):
        if i % 99 == 0:
            logging.fatal(f'Iteration {i}')
        if i % 10 == 0:
            logging.error(f'Iteration {i}')
        if i % 5 == 0:
            logging.warn(f'Iteration {i}')
        if i % 3 == 0:
            logging.info(f'Iteration {i}')
        if i % 2 == 0:
            logging.debug(f'Iteration {i}')
            logging.trace(f'Iteration {i}')

test_logging()