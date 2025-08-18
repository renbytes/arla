---
hide:
  - navigation
  - toc
---

# ARLA: The Agent Simulation Framework

<div class="hero-section" markdown>
<div class="hero-content" markdown>

## Build the Future of Agent-Based Modeling

ARLA combines cutting-edge cognitive architectures with high-performance simulation to create believable, intelligent agents that learn, adapt, and emerge complex behaviors.

[Get Started](tutorials/first-simulation.md){ .md-button .md-button--primary .md-button--stretch }
[View Documentation](guides/installation.md){ .md-button .md-button--stretch }

</div>
</div>

## Why ARLA?

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **High-Performance Core**

    ---

    Built on asynchronous Python with Entity-Component-System architecture. Scale to thousands of agents with concurrent execution and optimized memory management.

    [:octicons-arrow-right-24: Architecture Overview](architecture/index.md)

-   :material-brain:{ .lg .middle } **Cognitively-Rich Agents**

    ---

    Move beyond simple rules. Agents with memory, emotions, social awareness, and goal-driven behavior powered by Large Language Models.

    [:octicons-arrow-right-24: Cognitive Systems](api/agent-engine.md)

-   :material-puzzle:{ .lg .middle } **Modular & Extensible**

    ---

    Clean separation of data and logic through ECS. Add new behaviors, cognitive models, and environmental rules without touching the core engine.

    [:octicons-arrow-right-24: Developer Guide](developer/creating-actions.md)

-   :material-chart-line:{ .lg .middle } **Research-Ready**

    ---

    Built-in experiment management, MLflow integration, and comprehensive logging. Perfect for ablation studies and reproducible research.

    [:octicons-arrow-right-24: Running Experiments](guides/running-simulations.md)

</div>

## Quick Start

<div class="grid cards" markdown>

-   **1. Install**

    ---

    ```bash
    git clone https://github.com/renbytes/arla.git
    cd arla
    make setup && make up
    ```

-   **2. Run**

    ---

    ```bash
    make run-example
    ```

-   **3. Explore**

    ---

    Open [MLflow UI](http://localhost:5001) to view results and experiment tracking.

</div>

## Use Cases

<div class="grid cards" markdown>

-   :material-account-group:{ .lg .middle } **Social Dynamics**

    ---

    Study how societies form, cooperate, and conflict. Model everything from small groups to large populations.

-   :material-currency-usd:{ .lg .middle } **Economic Emergence**

    ---

    Watch markets, trade, and currency systems emerge naturally from agent interactions and resource scarcity.

-   :material-school:{ .lg .middle } **Learning & Adaptation**

    ---

    Research how agents learn from experience, form memories, and adapt their strategies over time.

-   :material-gavel:{ .lg .middle } **Moral Reasoning**

    ---

    Explore how ethical systems develop through social feedback and cultural transmission.

</div>

## Built for Researchers

=== "Academic"

    Perfect for computational social science, AI research, and complex systems studies. Built-in support for:
    
    - Reproducible experiments with configuration management
    - Statistical analysis with automated data collection
    - Publication-ready visualizations and metrics

=== "Industry"

    Prototype and test multi-agent systems for real-world applications:
    
    - Market simulation and economic modeling
    - Social network analysis and recommendation systems
    - Human-AI interaction studies

=== "Education"

    Teach complex systems, AI, and social dynamics with engaging simulations:
    
    - Pre-built scenarios for classroom use
    - Visual debugging and real-time monitoring
    - Comprehensive documentation and tutorials

## Community & Support

<div class="grid cards" markdown>

-   :fontawesome-brands-github:{ .lg .middle } **Open Source**

    ---

    MIT licensed with active development. Contribute features, report bugs, or extend the platform.

    [:octicons-arrow-right-24: Contributing Guide](contributing/setup.md)

-   :material-book-open:{ .lg .middle } **Documentation**

    ---

    Comprehensive guides, tutorials, and API reference. From first simulation to advanced cognitive architectures.

    [:octicons-arrow-right-24: Browse Docs](guides/installation.md)

-   :material-newspaper:{ .lg .middle } **Research Blog**

    ---

    Latest developments, research findings, and community showcases. Stay updated with the ARLA ecosystem.

    [:octicons-arrow-right-24: Read Blog](blog/index.md)

</div>

---

<div class="center-text" markdown>
**Ready to build intelligent agents?**

[Start Tutorial](tutorials/first-simulation.md){ .md-button .md-button--primary }
[Install ARLA](guides/installation.md){ .md-button }
</div>