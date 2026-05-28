# Cross-Lingual Active Learning Framework: Benchmarking Few-Shot Data Annotation

This repository contains the local prototyping engine built for the **Anote AI Research Fellowship** take-home assignment. The project explores a repeatable benchmarking framework for optimizing data annotation stability in low-resource contexts, using the Tulu language as a structural proxy.

## 🚀 Pillars Addressed
- **Data Curation:** High-signal labeling with minimal initial data inputs.
- **In-Context Optimization:** Evaluating semantic, syntactic, and inductive prompt functions: $P = f(T, C_x)$.
- **Active Learning Benchmarks:** Standardizing human-in-the-loop triggers using an algorithmic logic threshold ($\tau$).

## ⚙️ How It Works
The pipeline orchestrates a local LLM execution engine via **Ollama** (`llama3`) to execute stateless data classification across four distinct prompt strategies ($C_0, C_{lex}, C_{syn}, C_{ind}$). If the model's self-reflective `confidence_score` falls below a calibrated logic threshold, the architecture automatically triggers a simulated human intervention loop to protect data integrity.

## 📦 Quick Start
Running the Pipeline:
Simply execute the core script to observe the raw JSON output streams and framework execution scoring logs: python benchmark.py

### Prerequisites
Ensure you have the Ollama desktop application running locally:
```bash
pip install ollama
ollama pull llama3



```

