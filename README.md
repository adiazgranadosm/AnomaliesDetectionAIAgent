**Anomaly Detection AI Agent Framework**

**Abstract**

This repository presents a modular, AI‑driven framework for automating anomaly detection and response in industrial environments. The system integrates Large Language Models (LLMs), structured reasoning, and multi‑modal data processing to support end‑to‑end workflows including detection, classification, investigation, and corrective action. The framework is designed to be adaptable across heterogeneous industrial processes and to interface with existing organizational data sources, services, and operational tools.

**1. Introduction**
   
Anomaly detection in industrial systems is crucial for identifying defects, errors, and deviations that, if undetected, could lead to accidents, ecological damage, and economic losses. Traditional methods are rule-based, while modern approaches use AI and machine learning.

AI-powered automation is proposed to enhance anomaly response by integrating monitoring, analysis, decision-making, and corrective actions. The goal is to develop an AI agent framework optimized for detecting operational anomalies.
Large Language Models (LLMs) facilitate real-time responses by interpreting natural language prompts and coordinating that interact with databases, machine learning models, and external APIs.
Advancements in AI frameworks include Tools for external function invocation, Retrieval-Augmented Generation (RAG) for knowledge-based responses, and Long-Term Memory (LTM) for personalized interactions, further enhancing industrial AI applications.



**3. Objectives**
   
1) Design a flexible framework architecture for deploying an Artificial Intelligence (AI) agent to automate the end-to-end anomaly detection process, proposing an architecture that can be adapted for different types of processes and integrating existing technical components and services within a company.

2) Develop the framework in Python using APIs and libraries to streamline the process, ensuring maintainable code that minimizes the time required for modifications and updates. The framework executes key tasks in anomaly detection, including data analysis, identification and classification, information retrieval from both structured and unstructured data sources, and dynamic decision-making to facilitate preventive and corrective actions.

3) Validate the process by utilizing synthetic data, which can be generated to simulate various scenarios and assess the accuracy of the resulting responses and actions. This evaluation should include a comprehensive analysis of the quality of automated outputs, measuring key performance indicators such as response precision latency, and token utilization.  

4) Demonstrate that the implementation of this solution effectively mitigates or resolves risks and challenges inherent in industrial and corporate anomaly risks processes, thereby enhancing operational efficiency.

**3. System Overview**
   
<img width="800" alt="image" src="https://github.com/user-attachments/assets/69e5238e-ca5b-40d8-a3d3-b5697af97861" />

3.1 Inputs

- A user request in natural language to verify, extract information from multiple sources or apply an action related to addressing the anomaly.
- A CSV file from sensors or machinery to verify irregular patterns<img width="1443" height="90" alt="image" src="https://github.com/user-attachments/assets/a336b8fe-92e2-4afd-bc11-080f549bd18b" />

  
3.2 Task Execution

- The AI agent autonomously determines the workflow tasks required to address the anomaly.
- Each task is independent and uses different technologies to ensure effective resolution.
  
3.3 Outputs

- Tasks generate outputs such as findings, summaries, or service calls for for resolving the anomaly.
- The agent consolidates and delivers a natural language summary of the process and results to the user.
  

4. Architecture
   
<img width="600" alt="image" src="https://github.com/user-attachments/assets/5622d1eb-3327-449a-96ea-8364535acfcb" />


The orchestrator and routers constitute the core of the AI agent’s decision-making framework. 
This logic enables the agent to function autonomously, adapting the sequence and selection of operations according to the type and content of the request.
The main flow operates under a loop control mechanism.  
The agent system employs OpenAI's API as LLM and Pydantic library to define structured request and response types


**5. Experiments**
   
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








