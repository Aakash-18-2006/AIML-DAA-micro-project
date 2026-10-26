# 🧠 Student Study Path — AIML + DAA Micro Project
### AI-Based Personalized Study Sequence Optimization

A unified college micro-project combining **Artificial Intelligence & Machine Learning (AIML)** and **Design and Analysis of Algorithms (DAA)** to generate prerequisite-aware, personalized study recommendations for students.

---

## 📌 Introduction and Problem Statement

In academic computer science curricula, topics have strict dependency relationships (e.g., a student cannot properly master Dijkstra's algorithm without first understanding Graphs and Breadth-First Search).

Students frequently face several major learning roadblocks:
1. **Unidentified Bottlenecks**: Struggling with advanced topics because foundational prerequisite gaps were never diagnosed.
2. **Arbitrary Study Sequencing**: Studying topics in an inefficient or disordered sequence, skipping essential core fundamentals.
3. **Flawed Traditional Recommendation Systems**: Sorting topics purely by weakness or failure scores, completely disregarding prerequisite dependencies (producing impossible recommendations such as attempting Dijkstra before learning basic Graph representation).

**Student Study Path** solves this problem by uniting **Machine Learning Performance Prediction** with **DAA Directed Acyclic Graph (DAG) Sequencing**.

---

## 🎯 Project Objectives

1. **Student Metric Ingestion**: Capture student performance indicators (prior assessment scores, weekly practice hours, solved problem counts, and previous quiz attempts) for core Data Structures & Algorithms topics.
2. **Performance Estimation via AIML**: Estimate topic mastery percentage using Multiple Linear Regression and identify topics falling below a designated proficiency threshold (< 60%).
3. **Prerequisite Modeling via DAA Graphs**: Represent topic dependencies as a Directed Acyclic Graph (DAG) using a custom Adjacency List implementation (built without third-party graph libraries).
4. **Optimized Study Sequencing**: Apply **Kahn's Algorithm with a Greedy Priority Heuristic** to produce a valid topological study path that prioritizes the student's weakest topics first without ever violating prerequisite order.
5. **Interactive Exploration & Analytics**: Deliver a responsive Streamlit web dashboard providing input controls, performance summaries, dependency graph visualizations, and complexity analysis.

---

## ✨ Features and Functionality

- **Student Performance Metric Inputs**: Intuitive sliders and numeric controls for previous scores, practice hours, questions solved, and test attempts.
- **Continuous Topic Mastery Prediction**: Multi-variable linear regression model trained on realistic educational assessment data.
- **Automated Weak Topic Identification**: Configurable proficiency threshold (default 60%) to classify topics into "Weak" (needs study) and "Good" (proficient).
- **Prerequisite Graph Modeling**: Directed Acyclic Graph implemented from scratch with adjacency lists (`adj` and `rev_adj`).
- **Downstream Dependency Traversal (BFS)**: FIFO queue-based exploration to reveal all subsequent topics unlocked by mastering a given concept.
- **Upstream Prerequisite Resolution (DFS)**: Recursive traversal to discover all prerequisite ancestors required before tackling an advanced concept.
- **Greedy Topological Study Path Generation**: In-degree-based topological sequencing that dynamically prioritizes topics with lowest predicted scores when multiple prerequisites are met.
- **Dynamic Topic & Dependency Management**: Add, edit, or remove custom topics and dependencies dynamically with built-in Directed Acyclic Graph cycle validation and reliable local JSON file persistence (`data/subjects.json`) across page reloads and application restarts.
- **Interactive Web Interface**: Streamlit dashboard featuring progress indicators, status badges, metric cards, and step-by-step reasoning for each scheduled topic.
- **Automated Pipeline Testing**: Dedicated test suites (`test_pipeline.py` and `tests/test_subjects_persistence.py`) validating model accuracy, graph operations, BFS/DFS traversal, topological ordering, and file storage persistence.

---

## 🔗 AIML + DAA Integration Architecture

```text
                STUDENT PERFORMANCE DATA
          (Past Scores, Hours, Questions, Attempts)
                            ↓
                 [ AIML COMPONENT ]
             Multiple Linear Regression Model
              (Trained on student_data.csv)
                            ↓
                PREDICTED TOPIC SCORES (%)
                            ↓
                 WEAK TOPIC IDENTIFICATION
                   (Score < 60% Threshold)
                            ↓
                  [ DAA COMPONENT ]
              Topic Prerequisite DAG Graph
              (Custom Adjacency List Storage)
                            ↓
         DFS Ancestor Closure & BFS Downstream Traversal
                            ↓
            Greedy Topological Sort (Kahn's Logic)
         (Greedily picks lowest-scoring available topic)
                            ↓
          PERSONALIZED & PREREQUISITE-AWARE STUDY PATH
```

### Why this is a true integration:
* The **AIML regression model** dynamically evaluates student proficiency scores.
* The **DAA graph algorithm** utilizes these predicted scores as priority weights within a topological sort queue.
* The resulting recommendation is **neither pure ML nor static DAA**, but a hybrid schedule that is mathematically optimized and academically sound.

---

## 🤖 AI/ML Concepts Used

* **Supervised Learning**: Multiple Linear Regression ($y = \mathbf{w}^T\mathbf{x} + b$).
* **Features ($X$)**:
  * `previous_score`: Prior academic assessment score ($0 - 100$)
  * `practice_hours`: Dedicated weekly practice hours
  * `questions_solved`: Number of practice exercises completed
  * `previous_attempts`: Number of prior test/quiz attempts
* **Target ($y$)**: `target_score` (Estimated topic mastery percentage)
* **Model Training & Evaluation**:
  * Dataset split: 80% Training, 20% Testing
  * **Mean Absolute Error (MAE)**
  * **Mean Squared Error (MSE)**
  * **Coefficient of Determination ($R^2$ Score)**

---

## ⚡ Design and Analysis of Algorithms (DAA) Concepts Used

* **Graph Representation**: Directed Graph implemented from scratch using **Adjacency Lists** (`adj` and reverse adjacency list `rev_adj`). No external graph libraries (such as NetworkX) are used.
* **Breadth-First Search (BFS)** (`algorithms/bfs.py`):
  * Utilizes a FIFO Queue (`collections.deque`) to explore all downstream topics unlocked by a foundational concept layer-by-layer.
* **Depth-First Search (DFS)** (`algorithms/dfs.py`):
  * Recursively traverses upstream dependencies to retrieve the complete prerequisite chain required for any weak topic.
* **Cycle Detection**:
  * Uses recursive DFS with 3-state vertex coloring (unvisited, visiting, visited) to guarantee that prerequisite graphs remain strictly acyclic (DAG).
* **Study Path Optimization** (`algorithms/study_path.py`):
  * Constructs an induced subgraph containing all weak topics and their necessary prerequisite ancestors.
  * Employs **Kahn's Algorithm** with in-degree tracking.
  * **Greedy Choice**: When multiple topics have an in-degree of 0 (all prerequisites satisfied), the algorithm greedily dequeues the topic with the **lowest predicted score**, addressing urgent gaps as early as dependencies permit.

---

## ⏱️ Complexity Analysis

| Component / Algorithm | Time Complexity | Space Complexity | Description |
|---|---|---|---|
| **Multiple Linear Regression (Training)** | $O(n \cdot d^2 + d^3)$ | $O(d)$ | Closed-form OLS / SVD decomposition for $n$ samples and $d$ features |
| **Linear Regression (Inference)** | $O(d)$ | $O(1)$ | Scalar dot product per topic evaluation |
| **Breadth-First Search (BFS)** | $O(V + E)$ | $O(V)$ | Traverses downstream vertices and edges via FIFO Queue |
| **Depth-First Search (DFS)** | $O(V + E)$ | $O(V)$ | Traverses upstream prerequisite ancestors via call stack |
| **Cycle Detection** | $O(V + E)$ | $O(V)$ | DFS 3-color graph cycle validation |
| **Greedy Study Path Generation** | $O(V + E + V \log V)$ | $O(V + E)$ | Topological in-degree reduction with priority sorting |

*(Where $V$ = number of topics and $E$ = number of prerequisite dependencies).*

---

## 💻 Technologies and Libraries

- **Programming Language**: Python 3.9+ (compatible with Python 3.13 and 3.14)
- **Web Framework**: Streamlit (interactive web dashboard)
- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`linear_model.LinearRegression`, `model_selection.train_test_split`, `metrics`)
- **Data Visualization**: Matplotlib
- **Standard Library Modules**: `collections.deque`, `os`, `sys`
- **Version Control**: Git & GitHub

---

## 📂 Project Directory Structure

```text
AIML AND DAA MICRO/
│
├── data/
│   └── student_data.csv        # Realistic student assessment dataset
│
├── ml/
│   ├── preprocessing.py        # Dataset loading, cleaning, and train-test splitting
│   ├── train_model.py          # Linear regression training & evaluation metrics
│   └── predictor.py            # Model inference & weak topic classification
│
├── algorithms/
│   ├── graph.py                # Scratch adjacency-list directed graph & cycle detection
│   ├── bfs.py                  # FIFO queue-based downstream BFS traversal
│   ├── dfs.py                  # Recursive upstream DFS prerequisite chain solver
│   └── study_path.py           # Greedy topological sequence optimizer (Kahn's logic)
│
├── .streamlit/
│   └── config.toml             # Streamlit UI configuration
│
├── app.py                      # Interactive Streamlit Web Application
├── test_pipeline.py            # End-to-end verification and test script
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive project documentation
└── .gitignore                  # Git exclusion rules
```

---

## 🛠️ Topic Dependency Structure

```text
Arrays
 ├── Sorting
 └── Searching
      └── Dynamic Programming

Graphs
 ├── BFS
 │    └── Dijkstra
 ├── DFS
 └── Dijkstra
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure Python 3.9 or higher is installed on your system. Verify with:
```bash
python --version
# or on Windows:
py --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/Aakash-18-2006/AIML-DAA-micro-project.git
cd AIML-DAA-micro-project
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# or on Windows with py launcher:
py -3.13 -m pip install -r requirements.txt
```

---

## 💻 Commands to Run the Project

### Run the Interactive Web Dashboard:
```bash
streamlit run app.py
# or using Windows Python launcher:
py -3.13 -m streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

### Run the Verification & Test Pipeline:
```bash
python test_pipeline.py
# or using Windows Python launcher:
py -3.13 test_pipeline.py
```

---

## 🧪 Example Usage & Demonstration Scenarios

### Scenario 1: Graph Theory Bottleneck
* **Student Inputs**: Set `Graphs = 40%`, `BFS = 35%`, `Dijkstra = 25%`.
* **Behavior**: Despite `Dijkstra` having the lowest score ($25\%$), the system detects that `Graphs` and `BFS` are prerequisites.
* **Output Sequence**: `Graphs ➔ BFS ➔ Dijkstra`. Prerequisite constraints are strictly maintained.

### Scenario 2: Dynamic Programming Gap
* **Student Inputs**: Set `Dynamic Programming = 30%` and `Searching = 48%`.
* **Behavior**: The optimizer ensures `Arrays ➔ Searching` are scheduled prior to `Dynamic Programming`.
* **Output Sequence**: Addresses foundational array and searching prerequisites before tackling DP.

---

## ☁️ Deployment on Streamlit Community Cloud

The repository is fully configured for zero-configuration deployment on **Streamlit Community Cloud**:

1. **Repository**: Push this repository to GitHub (`main` branch).
2. **Dashboard**: Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. **New App Settings**:
   - **Repository**: `Aakash-18-2006/AIML-DAA-micro-project`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. **Deploy**: Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the application.

### 💾 Cloud Storage & Persistence Note
- **Local Environment**: Curriculum additions, edits, and deletions are saved directly to `data/subjects.json` and persist permanently across page refreshes and application restarts.
- **Streamlit Community Cloud Limitation**: Streamlit Community Cloud runs in ephemeral Docker containers. While `data/subjects.json` persists within a running container session, container sleep cycles, restarts, and new git deployments will reset local files to the repository's committed version.
- **Persistent Cloud Architecture Recommendation**: For multi-user, permanently persisted curriculum state in production cloud environments, replace local file I/O with an external cloud database service (such as **Supabase (PostgreSQL)**, **Firebase Firestore**, or **AWS DynamoDB**) with user authentication.

---

## ⚠️ Limitations and Future Enhancements

### Limitations
- **Linearity Assumption**: Multiple Linear Regression assumes predominantly linear relationships between practice metrics and topic scores.
- **Curriculum Scope**: Current default dependency graph is pre-configured for fundamental Data Structures and Algorithms topics.
- **Ephemeral Cloud Storage**: On free serverless tiers like Streamlit Community Cloud, local filesystem modifications do not persist across container reboots.

### Future Enhancements
- **Cloud Database Integration**: Integrate Supabase / PostgreSQL for permanent multi-user curriculum synchronization.
- **Non-Linear ML Models**: Introduce Random Forest Regressor or Gradient Boosted Decision Trees to capture non-linear student learning curves.
- **LMS / GitHub Integration**: Connect with Learning Management Systems (LMS) or GitHub classroom APIs to automatically fetch coding commits and quiz scores.
- **Adaptive Spaced Repetition**: Combine topological study path generation with spaced repetition intervals (e.g., Leitner system / SM-2 algorithm) for revision scheduling.
- **Multi-Subject Curriculum Graphs**: Expand the prerequisite graph engine to support cross-disciplinary curriculums (e.g., Mathematics, Operating Systems, Machine Learning).
