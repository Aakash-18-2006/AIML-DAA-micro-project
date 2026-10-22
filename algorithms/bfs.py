"""
Breadth-First Search (BFS) Module for Smart Study Path
Traverses topic dependencies layer by layer using a FIFO Queue.
Used to discover all downstream topics that depend on a given foundation topic.

Algorithm Characteristics:
  - Data Structure: FIFO Queue (collections.deque / list)
  - Time Complexity: O(V + E) where V is topics and E is prerequisite edges
  - Space Complexity: O(V) for visited set and queue
"""

from collections import deque
from typing import List, Dict, Any
from algorithms.graph import PrerequisiteGraph


def bfs_downstream_traversal(graph: PrerequisiteGraph, start_topic: str) -> List[str]:
    """
    Perform manual Breadth-First Search starting from 'start_topic'
    to discover all reachable downstream dependent topics in level order.

    Algorithm Steps:
      1. Initialize an empty queue and enqueue the start_topic.
      2. Mark start_topic as visited.
      3. While queue is not empty:
           a. Dequeue current topic.
           b. Add to traversal result list.
           c. For each dependent topic connected via directed edge (u -> v):
                If not visited, mark visited and enqueue.
      4. Return traversal order.
    """
    if start_topic not in graph.vertices:
        return []

    visited = set()
    queue = deque([start_topic])
    visited.add(start_topic)
    traversal_order: List[str] = []

    while queue:
        # Step 3a: Dequeue the front element (FIFO order)
        current = queue.popleft()
        traversal_order.append(current)

        # Step 3c: Explore all adjacent dependent topics
        for dependent in graph.get_dependents(current):
            if dependent not in visited:
                visited.add(dependent)
                queue.append(dependent)

    return traversal_order


def get_all_downstream_topics(graph: PrerequisiteGraph, start_topics: List[str]) -> List[str]:
    """
    Finds all unique topics reachable from a list of start topics using BFS.
    """
    all_downstream = set()
    for topic in start_topics:
        traversed = bfs_downstream_traversal(graph, topic)
        all_downstream.update(traversed)
    return list(all_downstream)
