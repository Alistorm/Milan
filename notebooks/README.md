## Notebooks

The `notebooks` folder contains Jupyter Notebooks used for experimentation, data analysis, and model development related to the Milan project.  These notebooks provide a record of the development process and allow for easy reproducibility of the results.

* **`data_exploration.ipynb`**: This notebook explores and visualizes the datasets used in the project, including the medical knowledge base and the multimodal training data. It provides insights into the data distribution, identifies potential biases, and informs data preprocessing steps.

* **`multimodal_model_training.ipynb`**: This notebook details the training process for the multimodal model. It includes code for data preprocessing, model architecture definition, hyperparameter tuning, and model evaluation. The specific model used (e.g., CNN, RNN, Transformer) and the training framework (e.g., TensorFlow, PyTorch) are described within the notebook.

* **`mistral_fine_tuning.ipynb`**: This notebook documents the fine-tuning process of the Mistral large language model using the `mistral_fine_tuning_data.jsonl` dataset.  It includes the code for loading the pre-trained Mistral model, preparing the data for fine-tuning, setting the training parameters, and evaluating the performance of the fine-tuned model.

* **`conversational_flow_design.ipynb`**: This notebook outlines the design and implementation of Milan's conversational flow. It includes experiments with different dialogue management strategies and the logic for handling user input and generating appropriate responses.

* **`evaluation_metrics.ipynb`**: This notebook defines and calculates the evaluation metrics used to assess Milan's performance, such as accuracy, precision, recall, F1-score for the medical knowledge component, and metrics like empathy scores, understanding scores for the multimodal component.  It also includes visualizations of the results.

* **`[Other notebooks]`**: [Describe any other notebooks included in the folder, such as notebooks for specific experiments, data preprocessing steps, or alternative model architectures.  Be specific about the purpose of each notebook.] For example:
    * **`sentiment_analysis_experiments.ipynb`**: This notebook explores different sentiment analysis techniques to enhance Milan's ability to recognize user emotions from text and voice input.



This detailed description of the `notebooks` folder helps to understand the development process and the different experiments conducted during the project.  Remember to adapt the content to accurately reflect the actual notebooks in your project.