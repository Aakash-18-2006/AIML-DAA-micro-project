"""
Graph Module for Smart Study Path
Manual Directed Graph implementation using Adjacency Lists.
Represents academic topics and their prerequisite dependency relationships.
No external graph libraries (like NetworkX) are used.
"""

from typing import Dict, List, Set


class PrerequisiteGraph:
    """
    A Directed Graph where:
      - Vertices (Nodes) represent academic topics.
      - A directed edge (u -> v) indicates topic 'u' is a prerequisite for topic 'v'.
        (i.e., 'u' must be learned before 'v').
    """

    def __init__(self):
        # Adjacency list: topic -> list of dependent topics (successors)
        self.adj: Dict[str, List[str]] = {}
        # Reverse adjacency list: topic -> list of prerequisite topics (predecessors)
        self.rev_adj: Dict[str, List[str]] = {}
        # Set of all registered vertices
        self.vertices: Set[str] = set()

    def add_vertex(self, topic: str) -> None:
        """Add a vertex to the graph if it doesn't already exist."""
        if topic not in self.vertices:
            self.vertices.add(topic)
            self.adj[topic] = []
            self.rev_adj[topic] = []

    def add_edge(self, prereq: str, dependent: str) -> None:
        """
        Add a directed edge from 'prereq' to 'dependent'.
        prereq -> dependent (prereq is a requirement for dependent).
        """
        self.add_vertex(prereq)
        self.add_vertex(dependent)

        if dependent not in self.adj[prereq]:
            self.adj[prereq].append(dependent)
        if prereq not in self.rev_adj[dependent]:
            self.rev_adj[dependent].append(prereq)

    def get_dependents(self, topic: str) -> List[str]:
        """Return all direct dependent topics (topics that require this topic)."""
        return self.adj.get(topic, [])

    def get_prerequisites(self, topic: str) -> List[str]:
        """Return direct prerequisite topics for a given topic."""
        return self.rev_adj.get(topic, [])

    def get_all_vertices(self) -> List[str]:
        """Return all vertices in deterministic sorted order."""
        return sorted(list(self.vertices))

    def get_in_degree(self, topic: str) -> int:
        """Return the number of prerequisites required for this topic."""
        return len(self.rev_adj.get(topic, []))

    def get_all_in_degrees(self) -> Dict[str, int]:
        """Return a mapping of all topics to their in-degrees."""
        return {v: len(self.rev_adj.get(v, [])) for v in self.vertices}

    def remove_vertex(self, topic: str) -> None:
        """Remove a vertex and all its connected in/out edges from the graph."""
        if topic in self.vertices:
            self.vertices.remove(topic)
            # Remove topic from dependents' prerequisite lists
            for dep in self.adj.get(topic, []):
                if topic in self.rev_adj.get(dep, []):
                    self.rev_adj[dep].remove(topic)
            # Remove topic from prerequisites' dependent lists
            for prereq in self.rev_adj.get(topic, []):
                if topic in self.adj.get(prereq, []):
                    self.adj[prereq].remove(topic)
            self.adj.pop(topic, None)
            self.rev_adj.pop(topic, None)

    def remove_edge(self, prereq: str, dependent: str) -> None:
        """Remove a directed edge between prereq and dependent."""
        if prereq in self.adj and dependent in self.adj[prereq]:
            self.adj[prereq].remove(dependent)
        if dependent in self.rev_adj and prereq in self.rev_adj[dependent]:
            self.rev_adj[dependent].remove(prereq)

    def get_all_edges(self) -> List[tuple]:
        """Return all directed edges as list of (prereq, dependent) tuples."""
        edges = []
        for u in sorted(self.vertices):
            for v in self.adj.get(u, []):
                edges.append((u, v))
        return edges

    def has_cycle(self) -> bool:
        """Checks if the graph contains any cycle using 3-color DFS."""
        visited = {}  # 0: unvisited, 1: visiting (in stack), 2: visited

        def dfs(node: str) -> bool:
            visited[node] = 1
            for neighbor in self.adj.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    return True
                if visited.get(neighbor, 0) == 0:
                    if dfs(neighbor):
                        return True
            visited[node] = 2
            return False

        for vertex in self.vertices:
            if visited.get(vertex, 0) == 0:
                if dfs(vertex):
                    return True
        return False


def build_default_study_graph() -> PrerequisiteGraph:
    """
    Construct the standard academic topic dependency graph for the micro-project:
      Arrays -> Sorting
      Arrays -> Searching
      Graphs -> BFS
      Graphs -> DFS
      BFS -> Dijkstra
      Graphs -> Dijkstra
      Searching -> Dynamic Programming
    """
    g = PrerequisiteGraph()

    edges = [
        ("Arrays", "Sorting"),
        ("Arrays", "Searching"),
        ("Graphs", "BFS"),
        ("Graphs", "DFS"),
        ("Graphs", "Dijkstra"),
        ("BFS", "Dijkstra"),
        ("Searching", "Dynamic Programming"),
    ]

    for prereq, dependent in edges:
        g.add_edge(prereq, dependent)

    return g


if __name__ == "__main__":
    graph = build_default_study_graph()
    print("Graph Vertices:", graph.get_all_vertices())
    for v in graph.get_all_vertices():
        print(f"Topic: {v:<20} | Prerequisites: {graph.get_prerequisites(v)} | Dependents: {graph.get_dependents(v)}")
