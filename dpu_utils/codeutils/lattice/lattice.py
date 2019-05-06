from collections import Counter
from functools import lru_cache
from itertools import chain
from typing import Iterable, List, Set, Optional, Union

from dpu_utils.utils.dataloading import save_json_gz, load_json_gz
from dpu_utils.mlutils import Vocabulary

__all__ = ['Lattice', 'LatticeVocabulary']


class Lattice:
    """Represents a lattice structure."""

    def __init__(self, elements: List[str], parent_relations: List[Set[int]]) -> None:
        self._elements = list(elements)
        self._element_to_id = {v: k for k, v in enumerate(self._elements)}
        self._parent_relations = [frozenset(parents) for parents in parent_relations]

    def __contains__(self, element: str) -> bool:
        return element in self._element_to_id

    @lru_cache(maxsize=1024)
    def parents(self, element: str) -> List[str]:
        """Get the parents of a given element"""
        if element not in self._element_to_id:
            return []

        element_id = self._element_to_id[element]
        all_parents = set()
        to_visit = list(self._parent_relations[element_id])
        while len(to_visit) > 0:
            next_element_id = to_visit.pop()
            all_parents.add(next_element_id)
            to_visit.extend(i for i in self._parent_relations[next_element_id] if i not in all_parents)

        return list(sorted(set(self._elements[i] for i in all_parents)))

    def to_dot(self, filename: str) -> None:
        with open(filename, 'w') as f:
            print('digraph G {', file=f)

            for i, element in enumerate(self._elements):
                print('n%s [label="%s"];' % (i, element), file=f)

            for i, parents in enumerate(self._parent_relations):
                for parent_id in parents:
                    print('n%s->n%s;' % (i, parent_id), file=f)

            print('}', file=f)  # digraph

    def save_as_json(self, filename: str) -> None:
        data = dict(types=self._elements, outgoingEdges=[list(p) for p in self._parent_relations])
        save_json_gz(data, filename)

    def merge(self, other_lattice: 'Lattice') -> None:
        self._parent_relations = [set(parents) for parents in self._parent_relations]  # Temporarily convert to sets
        for element in other_lattice._elements:
            if element not in self._element_to_id:
                self._element_to_id[element] = len(self._elements)
                self._elements.append(element)
                self._parent_relations.append(set())

        # Add parent relations
        for other_lattice_idx, element in enumerate(other_lattice._elements):
            for other_lattice_parent_idx in other_lattice._parent_relations[other_lattice_idx]:
                parent_name = other_lattice._elements[other_lattice_parent_idx]

                this_lattice_element_idx = self._element_to_id[element]
                this_lattice_parent_idx = self._element_to_id[parent_name]
                self._parent_relations[this_lattice_element_idx].add(this_lattice_parent_idx)
        self._parent_relations = [frozenset(parents) for parents in self._parent_relations]  # Reconvert to frozenset

    @staticmethod
    def load(filename: str) -> 'Lattice':
        types = load_json_gz(filename)
        return Lattice(types['types'], types['outgoingEdges'])


class LatticeVocabulary(Vocabulary):
    """A feature dictionary that instead of returning UNKs, closest parent element in a
    lattice"""

    def __init__(self, lattice: Lattice) -> None:
        super().__init__(True)
        self.__lattice = lattice

    def is_unk(self, token: str) -> bool:
        return token not in self.token_to_id

    @lru_cache(maxsize=512)
    def __get_list_of_implemented_types(self, token, alternative_lattice: Optional[Lattice] = None) -> List[int]:
        if token.startswith('type:'):
            if alternative_lattice is None:
                type_parents = self.__lattice.parents(token[len('type:'):])
            else:
                type_parents = alternative_lattice.parents(token[len('type:'):])
            return [self.token_to_id[t] for t in chain([token], ['type:' + p for p in type_parents]) if
                    t in self.token_to_id]
        if token in self.token_to_id:
            return [self.token_to_id[token]]
        return []

    def get_id_or_unk(self, token: str, alternative_lattice: Optional[Lattice] = None) -> List[int]:
        type_list = self.__get_list_of_implemented_types(token, alternative_lattice)
        if len(type_list) == 0:
            return [self.token_to_id[self.get_unk()]]
        return type_list

    def get_id_or_none(self, token: str, alternative_lattice: Optional[Lattice] = None) -> List[Optional[int]]:
        type_list = self.__get_list_of_implemented_types(token, alternative_lattice)
        if len(type_list) == 0:
            return [None]
        return type_list

    def add_batch_tokens(self, tokens: Iterable[str], lattice: Lattice, count_threshold: int = 5) -> None:
        token_counter = Counter(tokens)
        for token, count in list(token_counter.items()):
            if token.startswith('type:'):
                type_name = token[len('type:'):]
                for t in lattice.parents(type_name):
                    token_counter['type:' + t] += count
        for token, count in token_counter.items():
            if count >= count_threshold:
                self.add_or_get_id(token)

    @staticmethod
    def get_feature_dictionary_for(tokens: Iterable[str], lattice: Lattice,
                                   count_threshold: int = 5) -> 'LatticeVocabulary':
        """Deprecated: Use `get_vocabulary_for` instead."""
        return LatticeVocabulary.get_vocabulary_for(tokens, lattice, count_threshold)

    @staticmethod
    def get_vocabulary_for(tokens: Union[Iterable[str], Counter], lattice: Lattice,
                           count_threshold: int = 5, max_size: int = 100000) -> 'LatticeVocabulary':
        if type(tokens) is Counter:
            token_counter = tokens
        else:
            token_counter = Counter(tokens)
        for token, count in list(token_counter.items()):
            if token.startswith('type:'):
                type_name = token[len('type:'):]
                for t in lattice.parents(type_name):
                    token_counter['type:' + t] += count

        feature_dict = LatticeVocabulary(lattice)
        for token, count in token_counter.most_common(max_size):
            if count >= count_threshold:
                feature_dict.add_or_get_id(token)
        return feature_dict
