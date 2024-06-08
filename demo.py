from Customer_Churn.logger import logging
from Customer_Churn.exception import CustomerChurnException
import sys


try:
    a = 1/"10"
except Exception as e:
    logging.info(e)
    raise CustomerChurnException(e, sys) from e