from functools import lru_cache
from typing import List, Set

from .lattice import Lattice
from dpu_utils.utils.dataloading import load_json_gz


class CSharpLattice(Lattice):
    """Represents a lattice structure of C# types."""
    def __init__(self, elements: List[str], parent_relations: List[Set[int]]) -> None:
        super().__init__(elements, parent_relations)

    @lru_cache(maxsize=1024)
    def parents(self, element: str) -> List[str]:
        """Get the parents of a given element"""
        if element not in self._element_to_id:
            if element.endswith("[]"):
                inner_type = element[:-2]
                inner_type_parents = self.parents(inner_type)
                return list(sorted(set(inner_type_parent + "[]" for inner_type_parent in inner_type_parents)))

        return super().parents(element)

    @staticmethod
    def load(filename: str) -> 'CSharpLattice':
        types = load_json_gz(filename)
        return CSharpLattice(types['types'], types['outgoingEdges'])
