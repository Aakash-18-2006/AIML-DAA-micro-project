"""
Smart Study Path: AI-Based Personalized Study Sequence Optimization
Streamlit Web Application integrating AIML (Multiple Linear Regression) and DAA (Graph Algorithms & Greedy Optimization).
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import os
import json

import algorithms.graph
importlib.reload(algorithms.graph)

from ml.train_model import train_linear_regression
from ml.predictor import predict_all_topics, DEFAULT_WEAK_THRESHOLD
from algorithms.graph import build_default_study_graph, PrerequisiteGraph
from algorithms.bfs import bfs_downstream_traversal
from algorithms.dfs import dfs_prerequisite_traversal
from algorithms.study_path import generate_optimized_study_path

# Page configuration
st.set_page_config(
    page_title="Smart Study Path | AIML + DAA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern academic theme
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .badge-weak {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-good {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-prereq {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .step-badge {
        background-color: #3B82F6;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    /* Hide Streamlit Deploy button and related toolbar elements */
    .stAppDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stToolbarActions"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# Default curriculum configuration
DEFAULT_TOPICS = [
    "Arrays",
    "Sorting",
    "Searching",
    "Graphs",
    "BFS",
    "DFS",
    "Dijkstra",
    "Dynamic Programming",
]

DEFAULT_EDGES = [
    ("Arrays", "Sorting"),
    ("Arrays", "Searching"),
    ("Graphs", "BFS"),
    ("Graphs", "DFS"),
    ("Graphs", "Dijkstra"),
    ("BFS", "Dijkstra"),
    ("Searching", "Dynamic Programming"),
]

DEFAULT_METRICS = {
    "Arrays": {"previous_score": 75, "practice_hours": 12, "questions_solved": 35, "previous_attempts": 1},
    "Sorting": {"previous_score": 70, "practice_hours": 10, "questions_solved": 25, "previous_attempts": 1},
    "Searching": {"previous_score": 68, "practice_hours": 8, "questions_solved": 20, "previous_attempts": 1},
    "Graphs": {"previous_score": 40, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
    "BFS": {"previous_score": 35, "practice_hours": 3, "questions_solved": 6, "previous_attempts": 3},
    "DFS": {"previous_score": 42, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
    "Dijkstra": {"previous_score": 25, "practice_hours": 2, "questions_solved": 4, "previous_attempts": 4},
    "Dynamic Programming": {"previous_score": 62, "practice_hours": 9, "questions_solved": 22, "previous_attempts": 1},
}

# ----------------- RELIABLE SUBJECT PERSISTENCE (data/subjects.json) -----------------
SUBJECTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "subjects.json")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student_data.csv")


def load_subjects_data():
    """
    Loads subjects, prerequisite edges, and metrics from data/subjects.json.
    - If the file does not exist, creates it with DEFAULT_TOPICS, DEFAULT_EDGES, and DEFAULT_METRICS.
    - If the user has saved custom subjects (even an intentionally empty list []), respects that list without restoring defaults.
    - Safely handles missing, empty, or corrupted JSON files.
    """
    os.makedirs(os.path.dirname(SUBJECTS_FILE), exist_ok=True)
    
    if not os.path.exists(SUBJECTS_FILE):
        save_subjects_data(DEFAULT_TOPICS, DEFAULT_EDGES, DEFAULT_METRICS)
        return list(DEFAULT_TOPICS), list(DEFAULT_EDGES), {k: dict(v) for k, v in DEFAULT_METRICS.items()}
    
    try:
        with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                save_subjects_data(DEFAULT_TOPICS, DEFAULT_EDGES, DEFAULT_METRICS)
                return list(DEFAULT_TOPICS), list(DEFAULT_EDGES), {k: dict(v) for k, v in DEFAULT_METRICS.items()}
            
            data = json.loads(content)
            
            if isinstance(data, dict):
                subjects = data.get("subjects")
                edges = [tuple(e) for e in data.get("edges", [])]
                metrics = data.get("metrics", {})
                
                # If "subjects" key exists (even if empty list []), honor it!
                if subjects is not None and isinstance(subjects, list):
                    return subjects, edges, metrics
                
            elif isinstance(data, list):
                # Fallback support if file contains a plain list of subject strings
                return data, [e for e in DEFAULT_EDGES if e[0] in data and e[1] in data], {k: DEFAULT_METRICS.get(k, {}) for k in data}
    except Exception as e:
        print(f"[Warning] Error reading {SUBJECTS_FILE}: {e}. Safely falling back.")
    
    return list(DEFAULT_TOPICS), list(DEFAULT_EDGES), {k: dict(v) for k, v in DEFAULT_METRICS.items()}


def save_subjects_data(subjects, edges=None, metrics=None):
    """
    Safely writes subjects, edges, and metrics to data/subjects.json using UTF-8.
    """
    try:
        os.makedirs(os.path.dirname(SUBJECTS_FILE), exist_ok=True)
        payload = {
            "initialized": True,
            "subjects": list(subjects) if subjects is not None else [],
            "edges": [list(e) for e in (edges or [])],
            "metrics": dict(metrics or {})
        }
        with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Error] Failed to save {SUBJECTS_FILE}: {e}")
        return False


# Initialize session state from data/subjects.json on session start or page reload
if "topics" not in st.session_state:
    saved_topics, saved_edges, saved_metrics = load_subjects_data()
    st.session_state.topics = saved_topics
    st.session_state.edges = saved_edges
    st.session_state.topic_metrics = saved_metrics


@st.cache_resource
def get_ml_model():
    """Trains the Linear Regression model once."""
    try:
        model, metrics = train_linear_regression(DATASET_FILE)
        return model, metrics, None
    except Exception as e:
        return None, None, str(e)


def build_current_graph() -> PrerequisiteGraph:
    """Builds a PrerequisiteGraph from current session state topics and edges."""
    g = PrerequisiteGraph()
    for topic in st.session_state.topics:
        g.add_vertex(topic)
    for u, v in st.session_state.edges:
        if u in st.session_state.topics and v in st.session_state.topics:
            g.add_edge(u, v)
    return g


def check_graph_cycle(g: PrerequisiteGraph) -> bool:
    """Checks if the graph contains any cycle using 3-color DFS."""
    if hasattr(g, "has_cycle"):
        return g.has_cycle()
    
    visited = {}
    def dfs(node: str) -> bool:
        visited[node] = 1
        for neighbor in g.get_dependents(node):
            if visited.get(neighbor, 0) == 1:
                return True
            if visited.get(neighbor, 0) == 0:
                if dfs(neighbor):
                    return True
        visited[node] = 2
        return False

    for vertex in g.get_all_vertices():
        if visited.get(vertex, 0) == 0:
            if dfs(vertex):
                return True
    return False


def get_graph_edges(g: PrerequisiteGraph) -> list:
    """Returns all directed edges as (prereq, dependent) tuples."""
    if hasattr(g, "get_all_edges"):
        return g.get_all_edges()
    edges = []
    for u in g.get_all_vertices():
        for v in g.get_dependents(u):
            edges.append((u, v))
    return edges


# Initialize Model
model, metrics, load_error = get_ml_model()
graph = build_current_graph()

# Header
st.markdown("<h1 class='main-header'>🧠 Smart Study Path</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-Based Personalized Study Sequence Optimization (AIML + DAA Integration)</p>", unsafe_allow_html=True)

if load_error:
    st.error(f"Error loading model or dataset: {load_error}")
    st.stop()

# ----------------- SIDEBAR: CURRICULUM MANAGEMENT (ADD / EDIT / DELETE) -----------------
st.sidebar.markdown("### 📚 Subject & Topic Management")
st.sidebar.caption("💾 *Saved to `data/subjects.json`. Note: On Streamlit Community Cloud, instances are container-ephemeral; permanent multi-user cloud persistence across container restarts requires an external cloud database (e.g. Supabase / PostgreSQL).*")

# Expander 1: Add Topic
with st.sidebar.expander("➕ Add New Topic / Subject", expanded=False):
    new_topic_name = st.text_input("Topic Name", placeholder="e.g., Binary Trees, Greedy Methods", key="input_new_topic").strip()
    
    available_prereqs = [t for t in st.session_state.topics if t != new_topic_name]
    selected_prereqs = st.multiselect("Select Prerequisites (Optional)", available_prereqs, key="input_new_prereqs")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        new_prev_score = st.number_input("Past Score (%)", min_value=0, max_value=100, value=50, key="new_topic_score")
        new_practice_h = st.number_input("Practice Hours", min_value=0, max_value=100, value=6, key="new_topic_hours")
    with col_a2:
        new_q_solved = st.number_input("Questions Solved", min_value=0, max_value=200, value=15, key="new_topic_qs")
        new_attempts = st.number_input("Attempts", min_value=0, max_value=20, value=1, key="new_topic_att")
    
    if st.button("➕ Add Topic to Curriculum", use_container_width=True, type="primary"):
        if not new_topic_name:
            st.sidebar.error("⚠️ Please enter a topic name.")
        elif any(t.lower() == new_topic_name.lower() for t in st.session_state.topics):
            st.sidebar.error(f"⚠️ Topic '{new_topic_name}' already exists!")
        else:
            # Cycle detection before adding
            temp_graph = build_current_graph()
            temp_graph.add_vertex(new_topic_name)
            for p in selected_prereqs:
                temp_graph.add_edge(p, new_topic_name)
            
            if check_graph_cycle(temp_graph):
                st.sidebar.error("⚠️ Cannot add topic: Circular prerequisite dependency detected!")
            else:
                st.session_state.topics.append(new_topic_name)
                for p in selected_prereqs:
                    st.session_state.edges.append((p, new_topic_name))
                st.session_state.topic_metrics[new_topic_name] = {
                    "previous_score": int(new_prev_score),
                    "practice_hours": int(new_practice_h),
                    "questions_solved": int(new_q_solved),
                    "previous_attempts": int(new_attempts),
                }
                save_subjects_data(st.session_state.topics, st.session_state.edges, st.session_state.topic_metrics)
                st.toast(f"✅ Topic '{new_topic_name}' added and saved!")
                st.rerun()

# Expander 2: Edit Topic / Subject
with st.sidebar.expander("✏️ Edit Subject / Topic", expanded=False):
    if st.session_state.topics:
        edit_target = st.selectbox("Select Topic to Edit", st.session_state.topics, key="select_edit_topic")
        edit_name = st.text_input("Topic Name", value=edit_target, key="input_edit_name").strip()
        
        existing_prereqs = [u for u, v in st.session_state.edges if v == edit_target]
        available_edit_prereqs = [t for t in st.session_state.topics if t != edit_target]
        valid_default_prereqs = [p for p in existing_prereqs if p in available_edit_prereqs]
        edit_prereqs = st.multiselect("Edit Prerequisites", available_edit_prereqs, default=valid_default_prereqs, key="input_edit_prereqs")
        
        curr_m = st.session_state.topic_metrics.get(edit_target, {"previous_score": 50, "practice_hours": 6, "questions_solved": 15, "previous_attempts": 1})
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            e_score = st.number_input("Past Score (%)", min_value=0, max_value=100, value=int(curr_m.get("previous_score", 50)), key="edit_score")
            e_hours = st.number_input("Practice Hours", min_value=0, max_value=100, value=int(curr_m.get("practice_hours", 6)), key="edit_hours")
        with col_e2:
            e_qs = st.number_input("Questions Solved", min_value=0, max_value=200, value=int(curr_m.get("questions_solved", 15)), key="edit_qs")
            e_att = st.number_input("Attempts", min_value=0, max_value=20, value=int(curr_m.get("previous_attempts", 1)), key="edit_att")
        
        if st.button(f"💾 Save Changes to '{edit_target}'", use_container_width=True):
            if not edit_name:
                st.sidebar.error("⚠️ Topic name cannot be empty.")
            elif edit_name.lower() != edit_target.lower() and any(t.lower() == edit_name.lower() for t in st.session_state.topics):
                st.sidebar.error(f"⚠️ Topic '{edit_name}' already exists!")
            else:
                # Cycle detection
                temp_g = PrerequisiteGraph()
                for t in st.session_state.topics:
                    temp_g.add_vertex(edit_name if t == edit_target else t)
                for u, v in st.session_state.edges:
                    if v != edit_target:
                        src = edit_name if u == edit_target else u
                        temp_g.add_edge(src, v)
                for p in edit_prereqs:
                    src = edit_name if p == edit_target else p
                    temp_g.add_edge(src, edit_name)
                
                if check_graph_cycle(temp_g):
                    st.sidebar.error("⚠️ Cannot update: Circular prerequisite dependency detected!")
                else:
                    idx = st.session_state.topics.index(edit_target)
                    st.session_state.topics[idx] = edit_name
                    
                    new_edges = [(edit_name if u == edit_target else u, v) for u, v in st.session_state.edges if v != edit_target]
                    for p in edit_prereqs:
                        new_edges.append((p, edit_name))
                    st.session_state.edges = new_edges
                    
                    if edit_name != edit_target:
                        st.session_state.topic_metrics.pop(edit_target, None)
                    st.session_state.topic_metrics[edit_name] = {
                        "previous_score": int(e_score),
                        "practice_hours": int(e_hours),
                        "questions_solved": int(e_qs),
                        "previous_attempts": int(e_att),
                    }
                    save_subjects_data(st.session_state.topics, st.session_state.edges, st.session_state.topic_metrics)
                    st.toast(f"✅ Topic '{edit_name}' updated and saved!")
                    st.rerun()
    else:
        st.info("No topics available to edit.")

# Expander 3: Delete Topic & Reset
with st.sidebar.expander("🗑️ Delete Topic / Subject", expanded=False):
    if st.session_state.topics:
        topic_to_delete = st.selectbox("Select Topic to Delete", st.session_state.topics, key="delete_topic_select")
        
        if st.button(f"🗑️ Delete '{topic_to_delete}'", use_container_width=True):
            st.session_state.topics = [t for t in st.session_state.topics if t != topic_to_delete]
            st.session_state.edges = [(u, v) for u, v in st.session_state.edges if u != topic_to_delete and v != topic_to_delete]
            st.session_state.topic_metrics.pop(topic_to_delete, None)
            save_subjects_data(st.session_state.topics, st.session_state.edges, st.session_state.topic_metrics)
            st.toast(f"🗑️ Topic '{topic_to_delete}' deleted and removed from storage!")
            st.rerun()

        if st.button("🗑️ Clear All Topics (Empty Curriculum)", use_container_width=True):
            st.session_state.topics = []
            st.session_state.edges = []
            st.session_state.topic_metrics = {}
            save_subjects_data([], [], {})
            st.toast("🗑️ All topics cleared from curriculum and storage.")
            st.rerun()
    else:
        st.info("No topics available in curriculum.")

    if st.button("🔄 Reset to Default Curriculum", use_container_width=True):
        st.session_state.topics = list(DEFAULT_TOPICS)
        st.session_state.edges = list(DEFAULT_EDGES)
        st.session_state.topic_metrics = {k: dict(v) for k, v in DEFAULT_METRICS.items()}
        save_subjects_data(DEFAULT_TOPICS, DEFAULT_EDGES, DEFAULT_METRICS)
        st.toast("🔄 Reset to default 8 topics.")
        st.rerun()

st.sidebar.markdown("---")

# Sidebar: Student Profile Presets
st.sidebar.header("🎓 Student Profile & Input")

presets = {
    "Graph & Dijkstra Struggle": {
        "Arrays": {"previous_score": 75, "practice_hours": 12, "questions_solved": 35, "previous_attempts": 1},
        "Sorting": {"previous_score": 70, "practice_hours": 10, "questions_solved": 25, "previous_attempts": 1},
        "Searching": {"previous_score": 68, "practice_hours": 8, "questions_solved": 20, "previous_attempts": 1},
        "Graphs": {"previous_score": 40, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
        "BFS": {"previous_score": 35, "practice_hours": 3, "questions_solved": 6, "previous_attempts": 3},
        "DFS": {"previous_score": 42, "practice_hours": 4, "questions_solved": 8, "previous_attempts": 2},
        "Dijkstra": {"previous_score": 25, "practice_hours": 2, "questions_solved": 4, "previous_attempts": 4},
        "Dynamic Programming": {"previous_score": 62, "practice_hours": 9, "questions_solved": 22, "previous_attempts": 1},
    },
    "Dynamic Programming Gap": {
        "Arrays": {"previous_score": 80, "practice_hours": 14, "questions_solved": 40, "previous_attempts": 1},
        "Sorting": {"previous_score": 75, "practice_hours": 10, "questions_solved": 30, "previous_attempts": 1},
        "Searching": {"previous_score": 48, "practice_hours": 5, "questions_solved": 12, "previous_attempts": 2},
        "Graphs": {"previous_score": 70, "practice_hours": 12, "questions_solved": 28, "previous_attempts": 1},
        "BFS": {"previous_score": 68, "practice_hours": 10, "questions_solved": 24, "previous_attempts": 1},
        "DFS": {"previous_score": 72, "practice_hours": 11, "questions_solved": 26, "previous_attempts": 1},
        "Dijkstra": {"previous_score": 65, "practice_hours": 9, "questions_solved": 20, "previous_attempts": 1},
        "Dynamic Programming": {"previous_score": 30, "practice_hours": 3, "questions_solved": 6, "previous_attempts": 3},
    },
    "Comprehensive Foundations Needed": {
        "Arrays": {"previous_score": 45, "practice_hours": 5, "questions_solved": 12, "previous_attempts": 2},
        "Sorting": {"previous_score": 40, "practice_hours": 4, "questions_solved": 10, "previous_attempts": 3},
        "Searching": {"previous_score": 38, "practice_hours": 4, "questions_solved": 9, "previous_attempts": 3},
        "Graphs": {"previous_score": 35, "practice_hours": 3, "questions_solved": 8, "previous_attempts": 3},
        "BFS": {"previous_score": 30, "practice_hours": 3, "questions_solved": 6, "previous_attempts": 4},
        "DFS": {"previous_score": 32, "practice_hours": 3, "questions_solved": 7, "previous_attempts": 3},
        "Dijkstra": {"previous_score": 20, "practice_hours": 2, "questions_solved": 3, "previous_attempts": 4},
        "Dynamic Programming": {"previous_score": 25, "practice_hours": 2, "questions_solved": 4, "previous_attempts": 4},
    },
    "Custom Profile": None
}

selected_preset = st.sidebar.selectbox("Load Sample Student Profile", list(presets.keys()))

all_topics = st.session_state.topics
student_inputs = {}

if not all_topics:
    st.warning("⚠️ No topics in curriculum. Please add at least one topic from the sidebar or click 'Reset to Default Curriculum'.")
    st.stop()

# Expanders for topic inputs in sidebar
with st.sidebar.form("student_data_form"):
    st.subheader("Topic Performance Metrics")
    preset_data = presets.get(selected_preset) or {}

    for topic in all_topics:
        # Resolve defaults: from preset, or from session state topic_metrics, or fallback
        defaults = preset_data.get(topic) or st.session_state.topic_metrics.get(topic, {
            "previous_score": 50,
            "practice_hours": 6,
            "questions_solved": 15,
            "previous_attempts": 1,
        })

        is_expanded = (topic in ["Graphs", "BFS", "Dijkstra"] if selected_preset == "Graph & Dijkstra Struggle" else False)
        with st.expander(f"📘 {topic}", expanded=is_expanded):
            c1, c2 = st.columns(2)
            with c1:
                prev_score = st.number_input(
                    f"Previous Score (%)",
                    min_value=0,
                    max_value=100,
                    value=int(defaults["previous_score"]),
                    key=f"{topic}_score",
                )
                practice_h = st.number_input(
                    f"Practice Hours",
                    min_value=0,
                    max_value=100,
                    value=int(defaults["practice_hours"]),
                    key=f"{topic}_hours",
                )
            with c2:
                q_solved = st.number_input(
                    f"Questions Solved",
                    min_value=0,
                    max_value=200,
                    value=int(defaults["questions_solved"]),
                    key=f"{topic}_qs",
                )
                attempts = st.number_input(
                    f"Attempts",
                    min_value=0,
                    max_value=20,
                    value=int(defaults["previous_attempts"]),
                    key=f"{topic}_att",
                )

            student_inputs[topic] = {
                "previous_score": prev_score,
                "practice_hours": practice_h,
                "questions_solved": q_solved,
                "previous_attempts": attempts,
            }

    threshold = st.slider("Weak Topic Score Threshold (%)", min_value=40, max_value=80, value=60, step=5)
    include_all = st.checkbox("Include all topics in path (even strong topics)", value=False)
    submit_button = st.form_submit_button("🚀 Generate My Study Path", use_container_width=True)

if submit_button:
    for t_name, t_vals in student_inputs.items():
        st.session_state.topic_metrics[t_name] = dict(t_vals)
    save_subjects_data(st.session_state.topics, st.session_state.edges, st.session_state.topic_metrics)

# Main Dashboard Content
# Re-sync graph from current session state
graph = build_current_graph()

# Always compute initial state or update on submission
predictions = predict_all_topics(model, student_inputs, threshold=threshold)
study_path = generate_optimized_study_path(
    predicted_data=predictions,
    graph=graph,
    threshold=threshold,
    include_all_topics=include_all,
)

# Tabs layout
tab_results, tab_graph, tab_algorithms, tab_model = st.tabs([
    "🎯 Recommended Study Path & Predictions",
    "🕸️ Topic Dependency Graph & Inspector",
    "⚙️ Algorithm Analysis & Complexity",
    "📊 ML Model Evaluation",
])

# ----------------- TAB 1: RESULTS -----------------
with tab_results:
    # Top Metrics Row
    weak_count = sum(1 for p in predictions.values() if p["is_weak"])
    good_count = len(predictions) - weak_count

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Topics in Curriculum", len(all_topics))
    m2.metric("Weak Topics Detected (< " + str(threshold) + "%)", weak_count, delta=f"-{weak_count}" if weak_count > 0 else "0", delta_color="inverse")
    m3.metric("Strong / Good Topics", good_count, delta=f"+{good_count}")
    m4.metric("Study Steps in Optimized Path", len(study_path))

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1️⃣ Performance Prediction (AIML)")
        st.caption("Multiple Linear Regression estimates expected score based on past score, practice hours, solved problems, and attempts.")

        # Create styled table
        table_rows = []
        for topic in all_topics:
            p_data = predictions[topic]
            status_badge = "🔴 Weak" if p_data["is_weak"] else "🟢 Good"
            table_rows.append({
                "Topic": topic,
                "Predicted Score": f"{p_data['predicted_score']:.1f}%",
                "Status": status_badge,
                "Past Score": f"{student_inputs[topic]['previous_score']}%",
                "Practice (Hrs)": student_inputs[topic]['practice_hours'],
            })

        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("📈 Predicted Score Distribution")
        chart_df = pd.DataFrame({
            "Topic": all_topics,
            "Predicted Score (%)": [predictions[t]["predicted_score"] for t in all_topics],
        }).set_index("Topic")
        st.bar_chart(chart_df, color="#3B82F6")

    st.markdown("---")

    # Section 3: Recommended Study Path
    st.subheader("🎯 Section 3 — Recommended Study Path")
    st.caption("Generated using **Kahn's Topological Ordering** combined with a **Greedy Weakest-First Priority Heuristic**.")

    if not study_path:
        st.success("🎉 Excellent! All topics meet or exceed the performance threshold. No remedial study path required.")
    else:
        for item in study_path:
            is_weak = item["is_weak"]
            badge_class = "badge-weak" if is_weak else "badge-prereq"
            badge_text = "Weak Topic" if is_weak else "Prerequisite Requirement"

            prereqs_str = ", ".join(item["prerequisites"]) if item["prerequisites"] else "None (Foundation)"
            dependents_str = ", ".join(item["dependents"]) if item["dependents"] else "None (Terminal Topic)"

            st.markdown(f"""
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">
                        <span class="step-badge">{item['step']}</span> {item['topic']}
                    </div>
                    <div>
                        <span class="{badge_class}">{badge_text}</span>
                        <span style="font-weight: 700; margin-left: 8px; color: #1E293B;">Predicted: {item['predicted_score']}%</span>
                    </div>
                </div>
                <div style="color: #334155; margin-bottom: 6px; font-size: 0.95rem;">
                    💡 <strong>Reason:</strong> {item['reason']}
                </div>
                <div style="font-size: 0.85rem; color: #64748B;">
                    <strong>Prerequisites:</strong> {prereqs_str} &nbsp;|&nbsp; <strong>Unlocks:</strong> {dependents_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 2: GRAPH DEPENDENCIES & INSPECTOR -----------------
with tab_graph:
    st.subheader("🕸️ Section 2 — Topic Dependency Graph & Subject Info")
    st.caption("Directed Acyclic Graph (DAG) representing prerequisite relationships implemented with manual Adjacency Lists.")

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("#### Dynamic Adjacency List Structure")
        adj_data = []
        for vertex in graph.get_all_vertices():
            adj_data.append({
                "Topic (Vertex)": vertex,
                "Prerequisites (In-Edges)": ", ".join(graph.get_prerequisites(vertex)) or "None (Root)",
                "Dependents (Out-Edges)": ", ".join(graph.get_dependents(vertex)) or "None (Leaf)",
                "In-Degree": graph.get_in_degree(vertex),
            })
        st.dataframe(pd.DataFrame(adj_data), use_container_width=True, hide_index=True)

        st.markdown("#### Active Graph Summary")
        g_c1, g_c2, g_c3 = st.columns(3)
        g_c1.metric("Vertices (Topics)", len(graph.get_all_vertices()))
        g_c2.metric("Prerequisite Edges", len(get_graph_edges(graph)))
        g_c3.metric("Root Topics", sum(1 for v in graph.get_all_vertices() if graph.get_in_degree(v) == 0))

    with col_g2:
        st.markdown("#### 🔍 Interactive Subject & Topic Inspector")
        if graph.get_all_vertices():
            default_inspect_idx = 0
            if "BFS" in graph.get_all_vertices():
                default_inspect_idx = graph.get_all_vertices().index("BFS")
            
            inspect_topic = st.selectbox(
                "Select Subject / Topic to Inspect",
                graph.get_all_vertices(),
                index=default_inspect_idx,
                key="tab2_inspect_topic"
            )

            bfs_reach = bfs_downstream_traversal(graph, inspect_topic)
            dfs_ancestors = dfs_prerequisite_traversal(graph, inspect_topic)
            topic_pred = predictions.get(inspect_topic, {})
            topic_score = topic_pred.get("predicted_score", 0)
            topic_status = "🔴 Weak" if topic_pred.get("is_weak") else "🟢 Good"

            st.markdown(f"### Subject: `{inspect_topic}`")
            
            # Subject overview card
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"**Predicted Score:** `{topic_score:.1f}%` ({topic_status})")
                st.markdown(f"**Prerequisites (In-Degree):** `{graph.get_in_degree(inspect_topic)}`")
            with ic2:
                direct_prereqs = graph.get_prerequisites(inspect_topic)
                direct_deps = graph.get_dependents(inspect_topic)
                st.markdown(f"**Direct Prerequisites:** `{', '.join(direct_prereqs) if direct_prereqs else 'None'}`")
                st.markdown(f"**Direct Dependents:** `{', '.join(direct_deps) if direct_deps else 'None'}`")

            st.info(f"**Downstream Reachable Topics (BFS Traversal):**\n\n`{' ➔ '.join(bfs_reach)}`")
            
            prereq_chain = " ➔ ".join(reversed(dfs_ancestors)) if dfs_ancestors else "None (Foundation Topic)"
            st.warning(f"**Upstream Prerequisite Chain (DFS Traversal):**\n\n`{prereq_chain}`")
        else:
            st.info("No topics available in graph.")

# ----------------- TAB 3: ALGORITHM ANALYSIS -----------------
with tab_algorithms:
    st.subheader("⚙️ Section 4 — Algorithms Used & Integration Pipeline")

    st.markdown("""
    ```text
                    +--------------------------------+
                    |      Student Input Data        |
                    | (Past Score, Hours, Questions) |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |  Multiple Linear Regression    | (AIML Component)
                    |     (scikit-learn Model)       |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |   Predicted Topic Scores (%)   |
                    |  & Weak Topic Flagging (<60%)  |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |    Prerequisite DAG Graph      | (DAA Component)
                    |   (Adjacency List + DFS/BFS)   |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |  Greedy Topological Selection  | (Integration Hub)
                    | (Prerequisite-Aware Ordering)  |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |   Personalized Study Path      |
                    +--------------------------------+
    ```
    """)

    st.markdown("#### Algorithm Verification Checklist")
    st.markdown("""
    - ✅ **Multiple Linear Regression**: Evaluates continuous student attributes to predict expected score.
    - ✅ **Adjacency List Directed Graph**: Models prerequisites without third-party graph frameworks.
    - ✅ **Breadth-First Search (BFS)**: Explores downstream unlockable topics layer by layer using a FIFO queue.
    - ✅ **Depth-First Search (DFS)**: Resolves deep prerequisite ancestor chains recursively.
    - ✅ **Greedy Topological Sort (Kahn's Heuristic)**: Greedily selects the weakest available topic among currently unlocked nodes.
    """)

    st.subheader("⏱️ Section 5 — Theoretical Time & Space Complexity")
    complexity_data = [
        {
            "Algorithm": "Multiple Linear Regression (Training)",
            "Time Complexity": "O(n · d² + d³)",
            "Space Complexity": "O(d)",
            "Role & Description": "Fits optimal weights for d features across n training samples via Normal Equation / SVD.",
        },
        {
            "Algorithm": "Linear Regression (Inference / Prediction)",
            "Time Complexity": "O(d)",
            "Space Complexity": "O(1)",
            "Role & Description": "Calculates scalar score prediction per topic via dot product: y = W · X + b.",
        },
        {
            "Algorithm": "Breadth-First Search (BFS)",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Role & Description": "Explores dependent topics level by level via FIFO Queue.",
        },
        {
            "Algorithm": "Depth-First Search (DFS)",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Role & Description": "Traverses upstream prerequisite ancestors recursively.",
        },
        {
            "Algorithm": "Greedy Topological Study Path",
            "Time Complexity": "O(V + E + V log V)",
            "Space Complexity": "O(V + E)",
            "Role & Description": "Maintains in-degrees and greedily prioritizes weakest ready topics.",
        },
    ]
    st.dataframe(pd.DataFrame(complexity_data), use_container_width=True, hide_index=True)

# ----------------- TAB 4: MODEL EVALUATION -----------------
with tab_model:
    st.subheader("📊 Section 6 — Model Evaluation & Performance")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Mean Absolute Error (MAE)", f"{metrics['mae']}")
    col_e2.metric("Mean Squared Error (MSE)", f"{metrics['mse']}")
    col_e3.metric("R² Determination Score", f"{metrics['r2_score']}")

    st.markdown("#### Regression Model Parameters")
    coeff_df = pd.DataFrame({
        "Feature": ["Previous Score", "Practice Hours", "Questions Solved", "Previous Attempts"],
        "Learned Coefficient (Weight)": metrics["coefficients"],
    })
    st.dataframe(coeff_df, use_container_width=True, hide_index=True)
    st.info(f"**Model Intercept (Bias):** `{metrics['intercept']}`")

    st.warning("ℹ️ *Note: This model is a demonstration model trained on a small educational dataset for micro-project validation.*")
