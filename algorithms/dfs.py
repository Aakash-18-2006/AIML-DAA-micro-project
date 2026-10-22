"""
Depth-First Search (DFS) Module for Smart Study Path
Explores deep dependency chains to discover all upstream prerequisite topics
that must be completed before a student can study a target topic.

Algorithm Characteristics:
  - Traversal Strategy: Depth-First recursive exploration / LIFO Stack
  - Time Complexity: O(V + E) where V is topics and E is prerequisite links
  - Space Complexity: O(V) for call stack and visited tracking
"""

from typing import List, Set
from algorithms.graph import PrerequisiteGraph


def dfs_prerequisite_traversal(graph: PrerequisiteGraph, target_topic: str) -> List[str]:
    """
    Perform manual Depth-First Search traversing UPSTREAM (via rev_adj)
    to identify all prerequisite foundation topics required for 'target_topic'.

    Algorithm Steps:
      1. Use a visited set to avoid redundant processing or cycles.
      2. Recursively visit all immediate prerequisites (parents) of current topic.
      3. Append each prerequisite topic upon exploration.
      4. Return the list of required ancestors.
    """
    visited: Set[str] = set()
    prerequisites_order: List[str] = []

    def _dfs_helper(node: str):
        for parent in graph.get_prerequisites(node):
            if parent not in visited:
                visited.add(parent)
                _dfs_helper(parent)
                prerequisites_order.append(parent)

    if target_topic in graph.vertices:
        _dfs_helper(target_topic)

    return prerequisites_order


def get_all_required_prerequisites(graph: PrerequisiteGraph, target_topics: List[str]) -> Set[str]:
    """
    Returns the complete closure set of all upstream prerequisites for given target topics.
    """
    required = set()
    for topic in target_topics:
        ancestors = dfs_prerequisite_traversal(graph, topic)
        required.update(ancestors)
    return required
