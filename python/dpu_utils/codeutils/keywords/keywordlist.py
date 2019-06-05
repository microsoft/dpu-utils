import os
import keyword
from functools import lru_cache
from typing import FrozenSet

__all__ = ['get_language_keywords']


def load_file(name: str) -> FrozenSet[str]:
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return frozenset(l.strip() for l in f if len(l.strip()) > 0)


@lru_cache()
def get_language_keywords(language: str) -> FrozenSet[str]:
    """
    Returns the keywords of a programming language.

    There are some inconsistencies across languages wrt to
    what is considered a keyword. For example, the true/false
    literals are considered keywords in many languages. However,
    we exclude them here for consistency. We also exclude special
    functions-like keywords, such as `die()` in PHP.
    """
    if language == 'c':
        return load_file('c.txt')
    elif language == 'cpp' or language == 'c++':
        return load_file('cpp.txt')
    elif language == 'csharp' or language == 'c#':
        return load_file('csharp.txt')
    elif language == 'go':
        return load_file('go.txt')
    elif language == 'java':
        return load_file('java.txt')
    elif language == 'javascript':
        return load_file('javascript.txt')
    elif language == 'php':
        return load_file('php.txt')
    elif language == 'python':
        return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')
    elif language == 'ruby':
        return load_file('ruby.txt')
    elif language == 'typescript':
        return load_file('typescript.txt')
    else:
        raise Exception('Language %s not supported yet' % language)
