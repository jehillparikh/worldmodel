\documentclass{article}
\usepackage{amsmath, amssymb, graphicx, hyperref}
\title{\textbf{Image Encoders with DINO and DINOv2}}
\author{}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
Image encoders have evolved significantly with the emergence of **self-supervised learning** techniques. Recent advancements like **DINO** and **DINOv2** have demonstrated strong performance in learning feature representations without labeled data. In this post, we explore self-attention, transformers, and how they lead to advanced vision encoders.

\section{Self-Attention Mechanism}
Self-attention is a key component of transformer architectures and is responsible for capturing **long-range dependencies** in input sequences. It enables the model to assign different importance weights to various parts of the input, allowing for more dynamic feature extraction.

\subsection{Mathematical Formulation}
Given an input sequence of token embeddings $X$, we compute **Query (Q), Key (K), and Value (V)** matrices:
\begin{equation}
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
\end{equation}
where $W_Q$, $W_K$, and $W_V$ are learnable weight matrices. These matrices help in computing attention scores and refining feature representations.

The **scaled dot-product attention** is computed as:
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\end{equation}
where $d_k$ is the dimension of the key vectors, which stabilizes gradient updates.

\subsection{Multi-Head Attention}
Instead of computing a single attention function, transformers use **multi-head attention**, which allows the model to focus on different parts of the sequence simultaneously. The attention heads are computed as:
\begin{equation}
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h) W_O
\end{equation}
where each attention head computes attention independently before being concatenated and linearly transformed.

\section{Transformer Encoder Architecture}
The **Transformer encoder** consists of multiple stacked layers, each containing:
\begin{itemize}
    \item Multi-head self-attention mechanism
    \item Feed-forward network (FFN)
    \item Residual connections and Layer Normalization
\end{itemize}
Each encoder block follows this sequence:
\begin{enumerate}
    \item Input embeddings + Positional Encoding
    \item Multi-head self-attention
    \item Layer Normalization
    \item Feed-forward network
    \item Layer Normalization (again)
\end{enumerate}
The encoder processes the input in parallel, unlike RNNs, making it more efficient for long sequences.

\subsection{Masked Language Modeling (MLM)}
MLM is commonly used in natural language processing (NLP) tasks, where a fraction of the input tokens are randomly masked, and the model learns to predict the missing tokens based on their context. The objective function for MLM is:
\begin{equation}
\mathcal{L}_{MLM} = - \sum_{i \in M} \log P(x_i | x_{\backslash i})
\end{equation}
where $M$ represents the set of masked positions, and $P(x_i | x_{\backslash i})$ is the probability of correctly predicting the masked token.



\section{Vision Transformers (ViTs)}
ViTs apply **self-attention** to image data, treating images as sequences of patches rather than using convolutional operations.

\subsection{Patch Extraction and Encoding}
An image is split into patches, flattened, and passed through a **linear projection**:
\begin{equation}
X_{patch} = W_E \cdot X_{input} + b_E
\end{equation}
where $W_E$ is the embedding matrix and $b_E$ is a bias term. These embeddings are then processed using self-attention layers to extract contextual information.


\section{Contrastive Language-Image Pretraining (CLIP)}
CLIP is a multimodal self-supervised learning model designed to learn joint representations of images and text. It is trained on a large dataset of image-text pairs using a contrastive learning approach.

\subsection{Architecture}
CLIP consists of two main components:
\begin{itemize}
    \item An image encoder (ResNet or Vision Transformer)
    \item A text encoder (Transformer-based model)
\end{itemize}
Both encoders project images and text into a shared embedding space, where semantically similar images and text have higher similarity scores.

\subsection{Training Objective}
CLIP uses a contrastive learning objective where it maximizes the cosine similarity between matching image-text pairs while minimizing similarity between mismatched pairs:
\begin{equation}
\mathcal{L}_{CLIP} = - \sum_{i} \log \frac{e^{\cos(I_i, T_i)/\tau}}{\sum_j e^{\cos(I_i, T_j)/\tau}}
\end{equation}
where $I_i$ and $T_i$ are corresponding image-text pairs, and $\tau$ is a temperature parameter that controls the sharpness of the distribution.

\subsection{Applications of CLIP}
\begin{itemize}
    \item Zero-shot image classification
    \item Image retrieval and search
    \item Generating text-based descriptions of images
    \item Few-shot learning for downstream vision tasks
\end{itemize}
CLIP's ability to understand images and text jointly makes it a powerful model for many vision-language applications.




\section{DINO: Self-Supervised Vision Encoding}
DINO (Self-Distillation without Labels) is a **self-supervised learning method** that trains a **student** network to predict the output of a momentum **teacher** network.

\subsection{Key Features of DINO}
\begin{itemize}
    \item Works with both CNNs and ViTs
    \item Learns representations using **contrastive learning**
    \item Uses a momentum teacher-student setup
    \item Multi-crop augmentation for efficient feature learning
    \item Achieves competitive performance in unsupervised representation learning
\end{itemize}

\subsection{DINO Training Process}
DINO minimizes the cross-entropy loss between teacher and student outputs:
\begin{equation}
\mathcal{L}_{DINO} = - \sum_{i} p_i^T \log p_i^S
\end{equation}
where $p_i^T$ and $p_i^S$ are teacher and student predictions. The teacher network updates using an exponential moving average of the student’s weights.

\section{DINOv2: Improved Vision Encoding}
DINOv2 extends DINO with **Masked Image Modeling (MIM)** and **Self-Supervised Instance Discrimination**, improving its ability to capture both local and global features.

\subsection{Key Improvements in DINOv2}
\begin{itemize}
    \item Introduces **masked image modeling (MIM)** to learn local textures
    \item Uses **multi-crop augmentation** to enhance representation learning
    \item No reliance on **text supervision** (unlike CLIP)
    \item Improved transferability to **dense prediction tasks** (e.g., segmentation)
    \item Asymmetric teacher-student design for more effective training
    \item Incorporates KoLeo regularization to improve feature uniformity
\end{itemize}

\subsection{Masked Image Modeling (MIM)}
MIM extends the concept of MLM to images by randomly masking patches in an image and training the model to reconstruct the missing information. The objective function for MIM is:
\begin{equation}
\mathcal{L}_{MIM} = - \sum_{i \in M} \log P(I_i | I_{\backslash i})
\end{equation}
where $I_i$ represents the masked image patch, and $I_{\backslash i}$ denotes the observed patches.


\subsection{Training Objectives}
DINOv2 uses a **hybrid objective**:
\begin{equation}
\mathcal{L}_{DINOv2} = \lambda_1 \mathcal{L}_{MIM} + \lambda_2 \mathcal{L}_{Instance}
\end{equation}
where:
\begin{itemize}
    \item $\mathcal{L}_{MIM}$ is the loss for **masked image modeling**, focusing on reconstructing masked patches
    \item $\mathcal{L}_{Instance}$ is the loss for **instance discrimination**, ensuring robust feature alignment across augmentations
\end{itemize}
DINOv2 leverages a **vision-only training approach**, making it highly effective for self-supervised learning in image-based tasks.

\section{Conclusion}
DINO and DINOv2 represent significant advances in **self-supervised learning for vision tasks**. Their ability to learn rich representations **without labeled data** makes them valuable for diverse applications such as **image classification, retrieval, segmentation, and biomedical imaging**.

\end{document}
