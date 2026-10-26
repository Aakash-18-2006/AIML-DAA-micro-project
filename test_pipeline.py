"""
Test script to verify all ML and DAA components and integration pipeline.
"""
from ml.train_model import train_linear_regression
from ml.predictor import predict_all_topics
from algorithms.graph import build_default_study_graph
from algorithms.bfs import bfs_downstream_traversal
from algorithms.dfs import dfs_prerequisite_traversal
from algorithms.study_path import generate_optimized_study_path


def main():
    print("=== 1. Training ML Model ===")
    model, metrics = train_linear_regression("data/student_data.csv")
    print("Metrics:", metrics)

    print("\n=== 2. Building DAA Graph ===")
    graph = build_default_study_graph()
    print("All Vertices:", graph.get_all_vertices())

    print("\n=== 3. Testing BFS Downstream Traversal ===")
    bfs_result = bfs_downstream_traversal(graph, "Graphs")
    print("BFS reachable from 'Graphs':", bfs_result)
    assert "BFS" in bfs_result and "DFS" in bfs_result and "Dijkstra" in bfs_result

    print("\n=== 4. Testing DFS Upstream Prerequisite Traversal ===")
    dfs_result = dfs_prerequisite_traversal(graph, "Dijkstra")
    print("DFS prerequisite chain for 'Dijkstra':", dfs_result)
    assert "BFS" in dfs_result and "Graphs" in dfs_result

    print("\n=== 5. Predicting Student Performance ===")
    sample_inputs = {
        "Arrays": {"previous_score": 75, "practice_hours": 12, "questions_solved": 35, "previous_attempts": 1},
        "Sorting": {"previous_score": 70, "practice_hours": 10, "questions_solved": 25, "previous_attempts": 1},
        "Searching": {"previous_score": 68, "practice_hours": 8, "questions_solved": 20, "previous_attempts": 1},
        "Graphs": {"previous_score": 40, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
        "BFS": {"previous_score": 35, "practice_hours": 3, "questions_solved": 6, "previous_attempts": 3},
        "DFS": {"previous_score": 42, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
        "Dijkstra": {"previous_score": 25, "practice_hours": 2, "questions_solved": 4, "previous_attempts": 4},
        "Dynamic Programming": {"previous_score": 62, "practice_hours": 9, "questions_solved": 22, "previous_attempts": 1},
    }

    predictions = predict_all_topics(model, sample_inputs, threshold=60.0)
    for topic, info in predictions.items():
        print(f"Topic: {topic:<20} | Pred Score: {info['predicted_score']:<5} | Status: {info['status']}")

    print("\n=== 6. Generating Optimal Study Sequence (AIML + DAA) ===")
    study_path = generate_optimized_study_path(predictions, graph, threshold=60.0)
    for step in study_path:
        print(f"Step {step['step']}: {step['topic']:<12} (Score: {step['predicted_score']}%, Status: {step['status']}) - Reason: {step['reason']}")

    # Verification: Graphs must come before BFS and Dijkstra
    topics_in_order = [s["topic"] for s in study_path]
    print("\nGenerated Order:", topics_in_order)
    assert topics_in_order.index("Graphs") < topics_in_order.index("BFS")
    assert topics_in_order.index("BFS") < topics_in_order.index("Dijkstra")
    assert topics_in_order.index("Graphs") < topics_in_order.index("DFS")

    print("\n=== 7. Testing Dynamic Add & Delete Topics ===")
    graph.add_vertex("Binary Trees")
    graph.add_edge("Arrays", "Binary Trees")
    assert "Binary Trees" in graph.get_all_vertices()
    assert "Binary Trees" in graph.get_dependents("Arrays")
    assert "Arrays" in graph.get_prerequisites("Binary Trees")

    # Test cycle detection: DAG must return False
    assert not graph.has_cycle()

    # Introduce a cyclic dependency: Dijkstra -> Graphs (Graphs -> BFS -> Dijkstra -> Graphs)
    graph.add_edge("Dijkstra", "Graphs")
    assert graph.has_cycle(), "Graph must detect circular prerequisite dependency (Graphs -> BFS -> Dijkstra -> Graphs)!"
    graph.remove_edge("Dijkstra", "Graphs")
    assert not graph.has_cycle(), "Graph must return False after cycle edge is removed!"

    # Introduce direct 2-node cycle: Sorting -> Arrays (Arrays -> Sorting -> Arrays)
    graph.add_edge("Sorting", "Arrays")
    assert graph.has_cycle(), "Graph must detect circular dependency (Arrays -> Sorting -> Arrays)!"
    graph.remove_edge("Sorting", "Arrays")
    assert not graph.has_cycle()

    # Test deletion
    graph.remove_vertex("Binary Trees")
    assert "Binary Trees" not in graph.get_all_vertices()
    assert "Binary Trees" not in graph.get_dependents("Arrays")
    print("Dynamic add/delete & cycle verification passed.")

    print("\n[SUCCESS] ALL TESTS & ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
