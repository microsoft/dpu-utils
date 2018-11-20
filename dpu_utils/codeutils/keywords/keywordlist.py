import os
import keyword
from functools import lru_cache
from typing import List


def load_file(name: str) -> List[str]:
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return [l.strip() for l in f if len(l.strip()) > 0]


@lru_cache()
def get_language_keywords(language: str) -> List[str]:
    if language == 'c':
        return load_file('c.txt')
    elif language == 'csharp':
        return load_file('csharp.txt')
    elif language == 'go':
        return load_file('go.txt')
    elif language == 'java':
        return load_file('java.txt')
    elif language == 'javascript':
        return load_file('javascript.txt')
    elif language == 'python':
        return [k for k in keyword.kwlist if k != 'True' and k != 'False']
    else:
        raise Exception('Language %s not supported yet' % language)
