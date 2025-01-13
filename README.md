# Using CausaLM to Improve Interpretability for Clinical Notes
Current machine-learning approaches to medical diagnosis often rely on correlational patterns
between symptoms and diseases, risking misdiagnoses when symptoms are ambiguous or common across multiple conditions. In
this work, we move beyond correlation to investigate the causal influence of key symptoms—specifically “chest pain”—on diagnostic
predictions. Leveraging the CausaLM framework, we generate counterfactual text representations in which target concepts are effectively
“forgotten,” enabling a principled estimation of the causal effect of that concept on a model’s predicted disease distribution. By employing
Textual Representation-based Average Treatment Effect (TReATE), we quantify how the presence or absence of a symptom shapes
the model’s diagnostic outcomes, and contrast these findings against correlation-based baselines such as CONEXP. Our results offer deeper
insight into the decision-making behavior of clinical NLP models and can potentially inform more trustworthy, interpretable, and
causally-grounded decision-support tools in medical practice.
