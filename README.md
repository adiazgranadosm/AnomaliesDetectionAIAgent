Anomaly Detection AI Agent Framework
Abstract
This repository presents a modular, AI‑driven framework for automating anomaly detection and response in industrial environments. The system integrates Large Language Models (LLMs), structured reasoning, and multi‑modal data processing to support end‑to‑end workflows including detection, classification, investigation, and corrective action. The framework is designed to be adaptable across heterogeneous industrial processes and to interface with existing organizational data sources, services, and operational tools.

1. Introduction
   
Anomaly detection plays a critical role in industrial operations, where undetected deviations can lead to safety hazards, environmental damage, and significant economic losses. Traditional rule‑based systems often lack the flexibility required to handle complex, evolving operational conditions. Recent advances in artificial intelligence—particularly the emergence of LLM‑based agents—enable more adaptive, context‑aware, and autonomous anomaly‑management pipelines.
This project proposes an AI‑powered agent capable of orchestrating the full anomaly‑response lifecycle. The agent leverages natural‑language understanding, structured reasoning, and tool‑based function invocation to interpret user requests, analyze sensor data, retrieve relevant information, and execute targeted actions. The framework incorporates modern AI paradigms such as Retrieval‑Augmented Generation (RAG), external tool invocation, and long‑term memory mechanisms to enhance robustness and adaptability.

3. Objectives
   
The primary objectives of this work are:
- Architectural Design
Develop a flexible and extensible architecture for deploying an AI agent capable of automating anomaly detection workflows across diverse industrial processes. The architecture must integrate seamlessly with existing enterprise systems and services.
- Framework Development
Implement the architecture in Python using modern APIs and libraries to ensure maintainability and ease of extension. The agent must support data analysis, anomaly identification and classification, information retrieval from structured and unstructured sources, and dynamic decision‑making.
- Validation Using Synthetic Data
Evaluate the framework using synthetic datasets that simulate a range of operational scenarios. Performance is assessed through metrics such as response precision, latency, and token utilization.
- Demonstration of Practical Impact
Show that the proposed solution effectively mitigates operational risks and enhances the reliability of industrial anomaly‑management processes.
The agent follows a four‑stage operational workflow: Detection → Classification → Investigation → Action, adapting each stage to the organization’s available data and services.

3. System Overview
   
<img width="800" alt="image" src="https://github.com/user-attachments/assets/69e5238e-ca5b-40d8-a3d3-b5697af97861" />
3.1 Inputs
- Natural‑language user queries requesting verification, information extraction, or corrective actions.
- CSV files containing sensor or machinery data for anomaly detection.
3.2 Task Execution
- The agent autonomously determines the sequence of tasks required to address the anomaly.
- Each task is modular and may invoke different technologies (LLMs, APIs, data parsers, etc.).
3.3 Outputs
- Intermediate outputs include anomaly findings, summaries, and service calls.
- The agent synthesizes a final natural‑language explanation describing the full reasoning and results.

4. Architecture
   
<img width="600" alt="image" src="https://github.com/user-attachments/assets/5622d1eb-3327-449a-96ea-8364535acfcb" />
The architecture centers on two core components:
4.1 Orchestrator
Responsible for managing the global workflow, maintaining loop control, and coordinating interactions between modules.
4.2 Routers
Specialized decision units that determine which tools, models, or data sources should be invoked based on the content and intent of the request.
The system uses:
- OpenAI LLMs for reasoning and natural‑language interpretation
- Pydantic for defining structured request/response schemas
- Python‑based tools for data processing, retrieval, and action execution
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/3c418e6c-777b-4d5e-b158-027c2ef6f610" />

5. Experiments
   
A series of experiments were conducted using synthetic datasets to evaluate the agent’s performance across multiple anomaly scenarios.
<img width="400" alt="image" src="https://github.com/user-attachments/assets/dc6d8549-6653-4e28-b2a9-e6e121fdc921" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/194e0b72-79e8-44ab-8764-2d63919cccf2" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/0d5789b8-1c0a-4fb2-afae-97abac92b2d8" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/098a0f09-8f50-4dc3-ba7d-2888408ac59e" />
These experiments demonstrate the agent’s ability to:
- Detect irregular patterns in sensor data
- Classify anomalies based on contextual cues
- Retrieve relevant information from heterogeneous sources
- Produce coherent, actionable summaries








