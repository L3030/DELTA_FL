# DELTA: Diverse Client Sampling for Fasting Federated Learning
Official implementation of the NeurIPS 2023 paper "[DELTA: Diverse Client Sampling for Fasting Federated Learning](https://arxiv.org/abs/2205.13925)".
# Overview
**Abstract:** Partial client participation has been widely adopted in Federated Learning (FL) to reduce the communication burden efficiently. However, an inadequate client sampling scheme can lead to the selection of unrepresentative subsets, resulting in significant variance in model updates and slowed convergence. Existing sampling methods are either biased or can be further optimized for faster convergence. In this paper, we present DELTA, an unbiased sampling scheme designed to alleviate these issues. DELTA characterizes the effects of client diversity and local variance, and samples representative clients with valuable information for global model updates. In addition, DELTA is a proven optimal unbiased sampling scheme that minimizes variance caused by partial client participation and outperforms other unbiased sampling schemes in terms of convergence. Furthermore, to address full-client gradient dependence, we provide a practical version of DELTA depending on the available clients' information, and also analyze its convergence. Our results are validated through experiments on both synthetic and real-world datasets. 
