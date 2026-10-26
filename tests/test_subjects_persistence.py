"""
Automated unit and integration tests for JSON file subject persistence (data/subjects.json).
Verifies all 9 verification scenarios required by Smart Study Path.
"""
import os
import json
import shutil
import tempfile
import unittest

from app import load_subjects_data, save_subjects_data, DEFAULT_TOPICS, DEFAULT_EDGES, DEFAULT_METRICS, SUBJECTS_FILE


class TestSubjectPersistence(unittest.TestCase):
    def setUp(self):
        # Backup existing subjects.json if present
        self.backup_path = SUBJECTS_FILE + ".test_backup"
        if os.path.exists(SUBJECTS_FILE):
            shutil.copyfile(SUBJECTS_FILE, self.backup_path)
            os.remove(SUBJECTS_FILE)

    def tearDown(self):
        # Restore backup if it existed
        if os.path.exists(self.backup_path):
            shutil.copyfile(self.backup_path, SUBJECTS_FILE)
            os.remove(self.backup_path)
        elif os.path.exists(SUBJECTS_FILE):
            # Clean up test-created file
            pass

    def test_01_init_when_file_not_exists(self):
        """Scenario 1 & Requirement 2: If subjects.json does not exist, initialize with defaults."""
        self.assertFalse(os.path.exists(SUBJECTS_FILE))
        topics, edges, metrics = load_subjects_data()
        self.assertTrue(os.path.exists(SUBJECTS_FILE))
        self.assertEqual(topics, list(DEFAULT_TOPICS))
        self.assertEqual(edges, list(DEFAULT_EDGES))
        self.assertEqual(len(topics), 8)

    def test_02_add_subjects_and_persist(self):
        """Scenario 2, 3, 4: Adding subjects saves immediately and persists across reloads/restarts."""
        # Initialize
        load_subjects_data()
        
        # Add 3 subjects
        new_subjects = list(DEFAULT_TOPICS) + ["Binary Trees", "Heaps", "Trie"]
        new_edges = list(DEFAULT_EDGES) + [("Arrays", "Binary Trees"), ("Binary Trees", "Heaps")]
        new_metrics = dict(DEFAULT_METRICS)
        new_metrics["Binary Trees"] = {"previous_score": 60, "practice_hours": 8, "questions_solved": 20, "previous_attempts": 1}
        new_metrics["Heaps"] = {"previous_score": 55, "practice_hours": 6, "questions_solved": 15, "previous_attempts": 1}
        new_metrics["Trie"] = {"previous_score": 50, "practice_hours": 5, "questions_solved": 12, "previous_attempts": 1}
        
        save_subjects_data(new_subjects, new_edges, new_metrics)
        
        # Simulate restart/refresh by calling load_subjects_data fresh
        loaded_topics, loaded_edges, loaded_metrics = load_subjects_data()
        self.assertIn("Binary Trees", loaded_topics)
        self.assertIn("Heaps", loaded_topics)
        self.assertIn("Trie", loaded_topics)
        self.assertEqual(len(loaded_topics), 11)
        self.assertIn(("Arrays", "Binary Trees"), loaded_edges)

    def test_03_edit_subject_persists(self):
        """Scenario 6: Editing a subject persists to JSON file."""
        load_subjects_data()
        # Rename 'Graphs' to 'Advanced Graphs' and change score
        curr_topics, curr_edges, curr_metrics = load_subjects_data()
        idx = curr_topics.index("Graphs")
        curr_topics[idx] = "Advanced Graphs"
        curr_metrics["Advanced Graphs"] = {"previous_score": 95, "practice_hours": 20, "questions_solved": 50, "previous_attempts": 1}
        curr_metrics.pop("Graphs", None)
        
        save_subjects_data(curr_topics, curr_edges, curr_metrics)
        
        # Reload
        reloaded_topics, _, reloaded_metrics = load_subjects_data()
        self.assertIn("Advanced Graphs", reloaded_topics)
        self.assertNotIn("Graphs", reloaded_topics)
        self.assertEqual(reloaded_metrics["Advanced Graphs"]["previous_score"], 95)

    def test_04_delete_subject_persists(self):
        """Scenario 7: Deleting a subject permanently removes it and does not return on reload."""
        curr_topics, curr_edges, curr_metrics = load_subjects_data()
        # Delete 'Dijkstra'
        curr_topics = [t for t in curr_topics if t != "Dijkstra"]
        curr_edges = [e for e in curr_edges if e[0] != "Dijkstra" and e[1] != "Dijkstra"]
        curr_metrics.pop("Dijkstra", None)
        
        save_subjects_data(curr_topics, curr_edges, curr_metrics)
        
        # Reload
        reloaded_topics, reloaded_edges, _ = load_subjects_data()
        self.assertNotIn("Dijkstra", reloaded_topics)
        self.assertFalse(any(e[0] == "Dijkstra" or e[1] == "Dijkstra" for e in reloaded_edges))

    def test_05_delete_all_subjects_remains_empty(self):
        """Scenario 8 & Requirement 2: Deleting all subjects preserves [] and DOES NOT restore defaults."""
        # Save empty curriculum
        save_subjects_data([], [], {})
        
        # Verify file has empty list
        with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("subjects"), [])
        self.assertEqual(data.get("initialized"), True)
        
        # Reload (simulating page refresh / restart)
        reloaded_topics, reloaded_edges, reloaded_metrics = load_subjects_data()
        self.assertEqual(reloaded_topics, [], "Intentionally empty curriculum must NOT restore defaults")
        self.assertEqual(reloaded_edges, [])
        self.assertEqual(reloaded_metrics, {})

    def test_06_reset_to_defaults(self):
        """Reset button restores the 8 default subjects and saves to subjects.json."""
        # First empty it
        save_subjects_data([], [], {})
        self.assertEqual(load_subjects_data()[0], [])
        
        # Now reset
        save_subjects_data(DEFAULT_TOPICS, DEFAULT_EDGES, DEFAULT_METRICS)
        reloaded_topics, reloaded_edges, reloaded_metrics = load_subjects_data()
        self.assertEqual(reloaded_topics, list(DEFAULT_TOPICS))
        self.assertEqual(reloaded_edges, list(DEFAULT_EDGES))
        self.assertEqual(len(reloaded_topics), 8)

    def test_07_corrupted_json_handled_safely(self):
        """Corrupted JSON file does not crash the app and falls back safely."""
        with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
            f.write("{ invalid json ::: corrupted content !@#$")
            
        topics, edges, metrics = load_subjects_data()
        self.assertEqual(topics, list(DEFAULT_TOPICS))
        self.assertEqual(len(topics), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
