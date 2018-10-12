import sys
import traceback
import pdb
from typing import Callable

__all__= ['run_and_debug']


def run_and_debug(func: Callable[[], None], enable_debugging: bool)-> None:
    try:
        func()
    except:
        if enable_debugging:
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
