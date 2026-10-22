"""
Study Path Optimization Module for Smart Study Path
Integrates AIML predictions with DAA Graph Algorithms (Kahn's Topological Sort + Greedy Priority Selection).

Algorithm Pipeline:
  1. Input: Predicted scores from AIML Linear Regression model.
  2. Weak Topic Detection: Filter topics where predicted_score < threshold (default 60%).
  3. Prerequisite Resolution (DFS): Find all essential prerequisite topics needed for weak topics.
  4. Active Subgraph Construction: Induced graph of weak topics + prerequisite closure.
  5. Greedy Topological Ordering:
     - Maintains a Ready Queue of topics whose prerequisites are fully satisfied (in-degree = 0).
     - GREEDY CHOICE: Greedily select the topic with the LOWEST predicted score from the Ready Queue.
     - Decrements prerequisite in-degrees for dependents and repeats.
  6. Output: Step-by-step personalized study sequence that strictly honors prerequisites.
"""

from typing import Dict, List, Any, Set
import os
import sys
import heapq

# Ensure workspace root is in sys.path when executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from algorithms.graph import PrerequisiteGraph, build_default_study_graph
from algorithms.dfs import get_all_required_prerequisites


def generate_optimized_study_path(
    predicted_data: Dict[str, Dict[str, Any]],
    graph: PrerequisiteGraph = None,
    threshold: float = 60.0,
    include_all_topics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generates an optimal, prerequisite-aware study path guided by AIML predicted scores.

    Args:
        predicted_data: Dictionary mapping topic_name -> {'predicted_score': float, 'status': str, 'is_weak': bool}
        graph: PrerequisiteGraph instance (defaults to standard study graph)
        threshold: Score below which a topic is considered weak
        include_all_topics: If True, sequences all topics; If False, sequences weak topics + their prerequisites

    Returns:
        Ordered list of step dictionaries describing the optimal sequence.
    """
    if graph is None:
        graph = build_default_study_graph()

    # Step 1: Identify weak topics from AIML predictions
    weak_topics: Set[str] = {
        topic for topic, info in predicted_data.items()
        if info.get("predicted_score", 100.0) < threshold
    }

    # Step 2: If no weak topics found and not include_all, select lowest scoring topic or return empty
    if not weak_topics and not include_all_topics:
        # All topics are above threshold, but we can return an empty list or advice
        return []

    # Step 3: Determine target topics to include in the study sequence
    if include_all_topics:
        active_topics = set(graph.get_all_vertices()).union(set(predicted_data.keys()))
    else:
        # Resolve prerequisite ancestors using DFS
        prereq_ancestors = get_all_required_prerequisites(graph, list(weak_topics))
        active_topics = weak_topics.union(prereq_ancestors)

    # Step 4: Calculate in-degrees within the active topics subgraph
    in_degree: Dict[str, int] = {topic: 0 for topic in active_topics}
    for topic in active_topics:
        for prereq in graph.get_prerequisites(topic):
            if prereq in active_topics:
                in_degree[topic] += 1

    # Step 5: Initialize Ready List with all active topics having in-degree 0 (no unfulfilled prerequisites)
    ready_topics: List[str] = [topic for topic in active_topics if in_degree[topic] == 0]

    study_path: List[Dict[str, Any]] = []
    step_counter = 1

    # Step 6: Greedy Topological Processing
    while ready_topics:
        # --- GREEDY DECISION POINT ---
        # Among all topics whose prerequisites have already been mastered,
        # greedily pick the one with the LOWEST predicted score first.
        # This prioritizes critical academic weaknesses as early as dependencies allow.
        # Secondary tie-breaker: topic name for deterministic stability.
        ready_topics.sort(
            key=lambda t: (predicted_data.get(t, {}).get("predicted_score", 50.0), t)
        )
        selected_topic = ready_topics.pop(0)  # Pick the greedily chosen weakest ready topic

        topic_info = predicted_data.get(selected_topic, {})
        pred_score = topic_info.get("predicted_score", 50.0)
        is_weak = pred_score < threshold

        # Determine pedagogical reason for sequence placement
        direct_prereqs = [p for p in graph.get_prerequisites(selected_topic) if p in active_topics]
        direct_dependents = [d for d in graph.get_dependents(selected_topic) if d in active_topics]

        if is_weak:
            if direct_dependents:
                reason = f"Critical weak topic ({pred_score}%). Must be learned before advancing to {', '.join(direct_dependents)}."
            else:
                reason = f"Weak topic ({pred_score}%). Prerequisites completed; ready for focused practice."
        else:
            reason = f"Foundational prerequisite ({pred_score}%) required to unlock weak topic(s): {', '.join(direct_dependents)}."

        step_dict = {
            "step": step_counter,
            "topic": selected_topic,
            "predicted_score": pred_score,
            "status": "Weak" if is_weak else "Prerequisite Requirement",
            "is_weak": is_weak,
            "reason": reason,
            "prerequisites": graph.get_prerequisites(selected_topic),
            "dependents": graph.get_dependents(selected_topic),
        }
        study_path.append(step_dict)
        step_counter += 1

        # Decrement in-degree for downstream dependent active topics
        for dependent in graph.get_dependents(selected_topic):
            if dependent in active_topics:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready_topics.append(dependent)

    return study_path
