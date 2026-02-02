# Chapter 3: Transformers

The **Transformer** architecture, introduced in the landmark paper *"Attention Is All You Need"*, discarded recurrence and convolution entirely in favor of attention mechanisms. It has since become the foundation for virtually all state-of-the-art models in NLP, Vision, and beyond.

## The Architecture

The original Transformer is an **Encoder-Decoder** model designed for machine translation.

### The Encoder
The encoder processes the input sequence and compresses it into a context-rich representation. It consists of a stack of identical layers, each containing:
1.  **Multi-Head Self-Attention**
2.  **Feed-Forward Network (FFN)**
3.  **Layer Normalization & Residual Connections**

### The Decoder
The decoder generates the output sequence one token at a time. In addition to the components found in the encoder, it includes a **Masked Self-Attention** layer (to prevent looking at future tokens) and Cross-Attention (to attend to the encoder's output).

![Transformer Architecture](./Images/Transformer_arch.png "The Transformer Architecture")

## Evolution of Architectures

While the original model used both encoder and decoder, modern models often specialize:

| Type | Description | Examples | Best For |
| :--- | :--- | :--- | :--- |
| **Encoder-Only** | Bidirectional understanding of the whole input. | BERT, RoBERTa | Classification, NER, Sentiment Analysis |
| **Decoder-Only** | Unidirectional (left-to-right) generation. | GPT-4, Llama 3 | Text Generation, Chatbots, Code Completion |
| **Encoder-Decoder** | Maps input sequence to output sequence. | T5, BART | Translation, Summarization |

## Key Innovations & Updates

The standard Transformer has evolved significantly since 2017.

### Mixture of Experts (MoE)
As models grew larger, training costs skyrocketed. **Mixture of Experts (MoE)** (e.g., **Mixtral 8x7B**, **Switch Transformer**) addresses this by activating only a subset of parameters for each token.

*   **Sparse Activation**: Instead of using one dense FFN, the model has multiple "expert" networks (e.g., 8 experts). For every token, a **router network** (or gating network) selects the top-k experts (usually k=1 or 2) to process that specific token.
*   **Routing Mechanism**: The router computes a probability distribution over experts. The token is sent to the experts with the highest probabilities.
    $$ G(x) = \text{Softmax}(x \cdot W_g) $$
*   **Load Balancing**: A critical challenge is ensuring all experts are used equally. If one expert is popular, it becomes a bottleneck. Auxiliary loss functions are used to encourage balanced routing.
*   **Efficiency**: This allows models to have massive parameter counts (trillions) while keeping inference costs low (comparable to much smaller dense models).

### Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
Standard Multi-Head Attention (MHA) has a separate Key and Value head for every Query head. This results in a large KV cache size, slowing down inference.

*   **Multi-Query Attention (MQA)**: Uses **one** Key and Value head shared across **all** Query heads. This drastically reduces memory usage but can degrade performance.
*   **Grouped-Query Attention (GQA)**: A middle ground used in **Llama 2** and **Llama 3**. It groups Query heads (e.g., 8 groups) and assigns one Key/Value head to each group. This offers a better trade-off between performance and efficiency.

### Rotary Positional Embeddings (RoPE)
The original Transformer used absolute sinusoidal positional encodings. **RoPE** (Su et al., 2021) encodes relative position by rotating the query and key vectors in the embedding space. This allows models to generalize better to sequence lengths longer than those seen during training.

### ALiBi (Attention with Linear Biases)
ALiBi eliminates positional embeddings entirely by adding a static bias to the attention scores based on the distance between tokens. This simple change allows for extrapolation to much longer sequences.

### RMSNorm & SwiGLU
Modern LLMs (like Llama) often replace LayerNorm with **RMSNorm** (Root Mean Square Normalization) for stability and use **SwiGLU** activation functions in the FFN for better performance.

---

## LLM Agents

Large Language Models have evolved beyond simple text completion into autonomous **agents** capable of reasoning, planning, and taking actions in the world. This section explores the architecture and design patterns that enable LLM-based agents.

### What is an LLM Agent?

An LLM Agent extends a base language model with:

1. **Tools/Actions**: Ability to call external functions (search, code execution, APIs)
2. **Memory**: Short-term (context) and long-term (retrieval) storage
3. **Planning**: Breaking down complex tasks into subtasks
4. **Reasoning**: Chain-of-thought and self-reflection capabilities

```
┌─────────────────────────────────────────────┐
│                 LLM Agent                    │
├─────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │
│  │ Memory  │  │   LLM   │  │   Tools     │ │
│  │ - Short │◄─┤ (Brain) ├─►│ - Search    │ │
│  │ - Long  │  │         │  │ - Code      │ │
│  └─────────┘  └────┬────┘  │ - APIs      │ │
│                    │       └─────────────┘ │
│              ┌─────▼─────┐                  │
│              │ Planning  │                  │
│              │ & Action  │                  │
│              └───────────┘                  │
└─────────────────────────────────────────────┘
```

### Agent Architectures

#### ReAct (Reason + Act)

ReAct interleaves reasoning and action:

```
Thought: I need to find the population of Tokyo
Action: search("Tokyo population 2024")
Observation: Tokyo has approximately 14 million people...
Thought: Now I have the answer
Action: finish("Tokyo has ~14 million people")
```

The pattern: **Thought → Action → Observation → Thought → ...**

#### Plan-and-Execute

Separates planning from execution:

1. **Planner**: Creates a high-level plan of steps
2. **Executor**: Carries out each step
3. **Re-planner**: Adjusts plan based on results

#### Reflexion

Adds self-reflection and learning:

1. Execute task
2. Evaluate outcome
3. Reflect on what went wrong
4. Store reflection in memory
5. Retry with improved approach

### Tool Use

Modern agents can use diverse tools:

| Tool Type | Examples | Use Case |
|:----------|:---------|:---------|
| **Search** | Web search, RAG | Information retrieval |
| **Code** | Python interpreter, sandboxes | Computation, analysis |
| **APIs** | Weather, stocks, databases | Real-time data |
| **Files** | Read, write, edit | Document manipulation |
| **Browser** | Navigate, click, extract | Web automation |

#### Function Calling

Most LLM providers support structured tool use:

```json
{
  "name": "get_weather",
  "parameters": {
    "location": "San Francisco",
    "unit": "celsius"
  }
}
```

The model outputs structured JSON that can be parsed and executed.

### Multi-Agent Systems

Complex tasks benefit from multiple specialized agents:

- **Orchestrator**: Coordinates other agents
- **Researcher**: Gathers information
- **Coder**: Writes and debugs code
- **Critic**: Reviews and improves outputs

Frameworks like **AutoGen**, **CrewAI**, and **LangGraph** enable multi-agent workflows.

---

## Prompting Techniques

The way we communicate with LLMs dramatically affects their performance. Effective prompting is both an art and an emerging science.

### Basic Prompting Patterns

#### Zero-Shot Prompting

Direct instruction without examples:

```
Classify the sentiment of this review as positive or negative:
"The food was amazing but the service was slow."
```

#### Few-Shot Prompting

Provide examples to guide the model:

```
Classify sentiment:
"Great product!" → Positive
"Terrible experience" → Negative
"The food was amazing but the service was slow." →
```

#### Chain-of-Thought (CoT)

Encourage step-by-step reasoning:

```
Q: If John has 3 apples and buys 2 more, then gives
   away half, how many does he have?

A: Let me think step by step:
   1. John starts with 3 apples
   2. He buys 2 more: 3 + 2 = 5 apples
   3. He gives away half: 5 / 2 = 2.5 apples

   John has 2.5 apples (or 2 if we round down).
```

### Advanced Prompting Techniques

#### Self-Consistency

Generate multiple reasoning paths and vote on the answer:

1. Sample N different chain-of-thought responses
2. Extract the final answer from each
3. Return the most common answer (majority vote)

#### Tree of Thoughts (ToT)

Explore multiple reasoning branches:

```
Problem → [Branch A] → [Branch A1, A2]
       → [Branch B] → [Branch B1, B2]

Evaluate branches, prune bad ones, continue best paths
```

#### ReAct Prompting

Combine reasoning with actions:

```
Question: What is the elevation of the highest mountain
          in the country where Einstein was born?

Thought: I need to find where Einstein was born.
Action: Search[Albert Einstein birthplace]
Observation: Albert Einstein was born in Ulm, Germany.

Thought: Now I need the highest mountain in Germany.
Action: Search[highest mountain Germany]
Observation: The Zugspitze at 2,962 meters.

Thought: I have the answer.
Action: Finish[2,962 meters]
```

#### Prompt Chaining

Break complex tasks into sequential prompts:

```
Prompt 1: Extract key entities from this text
     ↓
Prompt 2: Research each entity
     ↓
Prompt 3: Synthesize findings into a report
```

### Prompting Best Practices

| Practice | Description |
|:---------|:------------|
| **Be Specific** | Clear, detailed instructions outperform vague ones |
| **Use Delimiters** | Separate sections with ```, ---, or XML tags |
| **Specify Format** | Request JSON, markdown, or structured output |
| **Provide Context** | Include relevant background information |
| **Use Personas** | "You are an expert data scientist..." |
| **Show Examples** | Few-shot learning improves consistency |
| **Request Reasoning** | "Explain your thinking" improves accuracy |

---

## Context Engineering

As LLMs are deployed in production, **context engineering** has emerged as a critical discipline—the systematic design of what information goes into the model's context window.

### The Context Window

Modern LLMs have varying context lengths:

| Model | Context Length |
|:------|:---------------|
| GPT-4 Turbo | 128K tokens |
| Claude 3 | 200K tokens |
| Gemini 1.5 | 1M+ tokens |
| Llama 3 | 8K-128K tokens |

Despite large windows, **what** goes in matters more than **how much**.

### Retrieval-Augmented Generation (RAG)

RAG dynamically retrieves relevant information:

```
┌──────────────┐     ┌─────────────┐     ┌─────────┐
│    Query     │────►│  Retriever  │────►│   LLM   │
└──────────────┘     │ (Vector DB) │     └─────────┘
                     └─────────────┘
                           │
                     ┌─────▼─────┐
                     │ Retrieved │
                     │ Documents │
                     └───────────┘
```

#### RAG Components

1. **Chunking**: Split documents into retrievable units
2. **Embedding**: Convert chunks to vectors
3. **Indexing**: Store in vector database
4. **Retrieval**: Find relevant chunks for query
5. **Generation**: LLM synthesizes answer from context

#### Advanced RAG Patterns

- **Hybrid Search**: Combine semantic + keyword search
- **Re-ranking**: Use cross-encoder to re-order results
- **Query Expansion**: Generate multiple query variants
- **HyDE**: Generate hypothetical document, then search

### Context Window Management

#### Strategies for Long Contexts

| Strategy | Description |
|:---------|:------------|
| **Summarization** | Compress older context into summaries |
| **Sliding Window** | Keep recent N tokens, summarize rest |
| **Hierarchical** | Multi-level summaries at different granularities |
| **Selective** | Only include task-relevant information |

#### The "Lost in the Middle" Problem

Research shows LLMs attend more to the **beginning** and **end** of context, often missing information in the middle. Solutions:

- Place critical information at start/end
- Use explicit references ("As mentioned in Document 3...")
- Break into multiple focused queries

### System Prompts & Instructions

The system prompt sets the agent's behavior:

```
You are a helpful coding assistant. You:
- Write clean, documented code
- Explain your reasoning
- Ask clarifying questions when requirements are unclear
- Suggest tests for your implementations

When you don't know something, say so clearly.
```

#### Instruction Hierarchy

1. **System Prompt**: Core behavior and constraints
2. **Retrieved Context**: Task-specific information
3. **Conversation History**: Prior exchanges
4. **User Message**: Current request

### Memory Systems

#### Short-Term Memory

The context window itself—limited but high-fidelity.

#### Long-Term Memory

External storage for persistent information:

- **Vector Databases**: Semantic search over past interactions
- **Knowledge Graphs**: Structured facts and relationships
- **Key-Value Stores**: Exact recall of specific information

#### Memory Architectures

```
┌─────────────────────────────────────────────┐
│              Working Memory                  │
│         (Current Context Window)            │
├─────────────────────────────────────────────┤
│   ┌───────────┐  ┌───────────┐  ┌────────┐ │
│   │ Episodic  │  │ Semantic  │  │ Procedural│
│   │ (Events)  │  │ (Facts)   │  │ (Skills) │ │
│   └───────────┘  └───────────┘  └────────┘ │
│              Long-Term Memory               │
└─────────────────────────────────────────────┘
```

### Structured Output

Constrain model outputs for reliability:

#### JSON Mode

```
Return your analysis as JSON:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0-1.0,
  "key_phrases": ["phrase1", "phrase2"]
}
```

#### Schema Enforcement

Tools like **Instructor**, **Outlines**, and **Guidance** guarantee valid structured outputs through constrained decoding.

### Evaluation & Iteration

Context engineering requires systematic evaluation:

1. **Define Metrics**: Accuracy, latency, cost, user satisfaction
2. **Create Test Sets**: Representative queries with expected outputs
3. **A/B Test**: Compare prompt/context variations
4. **Monitor Production**: Track real-world performance
5. **Iterate**: Continuously improve based on data

---

## Agentic Frameworks & Tools

Several frameworks have emerged to build LLM applications:

| Framework | Focus | Key Features |
|:----------|:------|:-------------|
| **LangChain** | General-purpose | Chains, agents, extensive integrations |
| **LlamaIndex** | RAG & data | Document loaders, indices, query engines |
| **AutoGen** | Multi-agent | Conversational agents, code execution |
| **CrewAI** | Multi-agent | Role-based agents, workflows |
| **Semantic Kernel** | Enterprise | Microsoft ecosystem, planners |
| **DSPy** | Optimization | Programmatic prompt optimization |

### Building Production Agents

Key considerations for production deployment:

1. **Reliability**: Handle failures gracefully, retry logic
2. **Observability**: Log all LLM calls, trace execution
3. **Cost Control**: Token budgets, caching, model routing
4. **Safety**: Input/output filtering, rate limiting
5. **Latency**: Streaming, parallel tool calls, caching

---

## Theoretical Insight: LLMs as World Models

A foundational paper by Richens et al. (ICML 2025), **"General Agents Contain World Models"**, provides crucial theoretical grounding for understanding why transformer-based agents are so effective.

### The Core Theorem

The paper proves that any agent capable of generalizing to multi-step, goal-directed tasks **must have learned a world model**—an internal representation that predicts how actions affect future states.

> **Theorem**: General goal-directed behavior is impossible without predictive modeling of environment dynamics.

### Why This Matters for Transformers

#### Next-Token Prediction IS World Modeling

When an LLM is trained to predict the next token, it implicitly learns:

$$
P(x_{t+1} | x_{1:t}) \approx P(\text{future} | \text{past})
$$

For text describing the world, this becomes a **world model**:

- "If I drop a ball, it will..." → Physics prediction
- "If the economy contracts, then..." → Economic dynamics
- "After mixing the chemicals..." → Causal reasoning

#### Emergent Capabilities Explained

The theorem explains why capabilities emerge with scale:

| Capability | World Model Interpretation |
|:-----------|:---------------------------|
| **Chain-of-thought** | Explicit simulation using world model |
| **In-context learning** | Rapid world model adaptation |
| **Reasoning** | Multi-step prediction through states |
| **Planning** | Evaluating action consequences |

### Implications for Agent Design

#### 1. World Model Quality Bounds Capability

$$
\text{Agent Capability} \leq f(\text{World Model Accuracy})
$$

To build more capable agents, we must improve their world models:
- Train on more diverse data (better coverage)
- Use larger models (more capacity)
- Employ explicit world modeling objectives

#### 2. Failures are World Model Failures

When LLM agents fail, it's often because their world model is wrong:

| Failure Mode | World Model Deficiency |
|:-------------|:----------------------|
| **Hallucinations** | Incorrect predictions about facts |
| **Planning failures** | Wrong action-consequence mapping |
| **Physical errors** | Insufficient physics modeling |
| **Social mistakes** | Inaccurate theory of mind |

#### 3. Extracting and Analyzing World Models

The paper shows world models can be **extracted** from trained agents:

```
Agent Policy → Probe Representations → Decode Predictions
                                            ↓
                         Extracted World Model
```

This enables:
- **Debugging**: Find where the world model is wrong
- **Alignment**: Verify beliefs match reality
- **Improvement**: Target training on model weaknesses

### Connecting to RAG and Tool Use

#### RAG as External World Model

Retrieval-Augmented Generation can be viewed as extending the agent's world model:

```
Internal World Model (LLM parameters)
           +
External World Model (Retrieved documents)
           =
Augmented World Model (More accurate predictions)
```

#### Tools as World Model Extensions

Tools extend what the agent can predict:

- **Calculator**: Precise numerical predictions
- **Search**: Access to factual world state
- **Code execution**: Complex logical predictions
- **APIs**: Real-time world information

### The Model-Free Illusion

Even "model-free" approaches to LLM agents implicitly rely on world models:

- **Zero-shot prompting**: Uses pre-trained world model
- **Few-shot learning**: Adapts world model with examples
- **Fine-tuning**: Specializes world model for domain

The paper proves there is no escaping this: **effective agents require world models**, whether explicit or implicit.

### Practical Takeaways

For building better LLM agents:

1. **Invest in world model quality**: Better base models, more diverse training
2. **Augment with retrieval**: Extend world model with external knowledge
3. **Use tools strategically**: Fill gaps in world model coverage
4. **Analyze failures**: Identify and fix world model deficiencies
5. **Evaluate world models**: Measure predictive accuracy, not just task performance

This theoretical foundation helps explain why current approaches work and guides the development of more capable AI agents.
