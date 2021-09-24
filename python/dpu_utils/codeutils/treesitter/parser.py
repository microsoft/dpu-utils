import logging
import os
from tempfile import TemporaryDirectory

from tree_sitter import Language, Parser

__all__ = ["parser_for"]

_LANGUAGE_REPOS = {
    "bash": ("https://github.com/tree-sitter/tree-sitter-bash", ""),
    "c": ("https://github.com/tree-sitter/tree-sitter-c", ""),
    "c_sharp": ("https://github.com/tree-sitter/tree-sitter-c-sharp", ""),
    "css": ("https://github.com/tree-sitter/tree-sitter-css", ""),
    "cpp": ("https://github.com/tree-sitter/tree-sitter-cpp", ""),
    "html": ("https://github.com/tree-sitter/tree-sitter-html", ""),
    "go": ("https://github.com/tree-sitter/tree-sitter-go", ""),
    "java": ("https://github.com/tree-sitter/tree-sitter-java", ""),
    "javascript": ("https://github.com/tree-sitter/tree-sitter-javascript", ""),
    "julia": ("https://github.com/tree-sitter/tree-sitter-julia", ""),
    "php": ("https://github.com/tree-sitter/tree-sitter-php", ""),
    "python": ("https://github.com/tree-sitter/tree-sitter-python", ""),
    "ruby": ("https://github.com/tree-sitter/tree-sitter-ruby", ""),
    "rust": ("https://github.com/tree-sitter/tree-sitter-rust", ""),
    "scala": ("https://github.com/tree-sitter/tree-sitter-scala", ""),
    "typescript": ("https://github.com/tree-sitter/tree-sitter-typescript", "typescript"),
}

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "build", "treesitter-lib.so")
TREE_SITTER_LANG_VER = "v0.19.0"

if not os.path.exists(LIBRARY_DIR):
    logging.warning("TreeSitter has not been compiled. Cloning languages and building...")
    from git import Repo
    with TemporaryDirectory() as dir:
        # Clone all repos above at the given tag
        repo_dirs = []
        for lang, (url, suffix) in _LANGUAGE_REPOS.items():
            logging.warning(f"Cloning `{lang}`...")
            repo_dir = os.path.join(dir, lang)
            repo = Repo.clone_from(url, repo_dir)
            repo.git.checkout(TREE_SITTER_LANG_VER)
            repo_dirs.append(os.path.join(repo_dir, suffix))

        # Build library by pointing to each repo
        logging.warning(f"Building Tree-sitter Library...")
        Language.build_library(LIBRARY_DIR, repo_dirs)

_LANGUAGES = {}
for language in _LANGUAGE_REPOS:
    _LANGUAGES[language] = Language(LIBRARY_DIR, language)

# Add aliases
_ALIASES = {
    "c++": "cpp",
    "c#": "c_sharp",
    "csharp": "c_sharp",
    "js": "javascript",
    "ts": "typescript"
}
for alias, target in _ALIASES.items():
    _LANGUAGES[alias] = _LANGUAGES[target]

def parser_for(language: str) -> Parser:
    parser = Parser()
    parser.set_language(_LANGUAGES[language])
    return parser
