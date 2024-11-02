# macaw-icliniq-medical-chatbot

Project 5 - Bias and Harmful Content in Conversational Search

Group 29: A. C., R. C. and T. vD.

---

### **Proposal:** Healthcare Information via Conversational Agents

#### **Motivation**
Healthcare conversational agents are increasingly relied upon for medical advice. However, these systems may reflect biases or generate harmful misinformation, potentially affecting certain demographic groups disproportionately. Biases in gender, race, socioeconomic status, or stigmatized conditions could lead to unequal treatment recommendations, putting users' health at risk. For example, women are often underdiagnosed for certain conditions, such as heart disease, because their pain is attributed to emotional causes rather than clinical ones. This project aims to investigate gender biases in healthcare advice provided by conversational agents, with a focus on pain management, a domain where such biases are well-documented.

---

### **Research Question**
Do conversational healthcare agents provide gender-biased responses in pain management scenarios, particularly regarding:
- Pain severity
- Diagnosis
- Emotional response
- Detail level
- Treatment recommendations?

#### **Hypotheses**
We propose the following hypotheses:
- **Hypothesis 1:** Women are more likely to have their pain minimized or attributed to emotional factors in conversational agent responses compared to men.
- **Hypothesis 2:** Men receive more clinical or solution-oriented advice, while women receive more emotionally supportive or dismissive responses.
- **Hypothesis 3:** Gender bias in pain management manifests through differences in language tone, recommended treatments, and degree of empathy shown.

---

### **Resources**
- **Fine-tuned Data**: Healthcare-related conversational logs, dataset on healthcare advice; the iCliniq Dataset, iCliniq Medical QA 2 (from Kaggle).
- **Software**: Macaw, for simulating healthcare interactions.
- **Users**: Simulated healthcare queries from diverse demographic backgrounds.

#### **Additional Literature**
- Samulowitz, A., Gremyr, I., Eriksson, E., & Hensing, G. (2018). “Brave Men” and “Emotional Women”: A Theory-Guided Literature Review on Gender Bias in Health Care and Gendered Norms towards Patients with Chronic Pain. *Pain Research And Management, 2018*, 1–14. [https://doi.org/10.1155/2018/6358624](https://doi.org/10.1155/2018/6358624)
- LeResche, L. (2011). Defining gender disparities in pain management. *Clinical Orthopaedics And Related Research, 469(7)*, 1871–1877. [https://doi.org/10.1007/s11999-010-1759-9](https://doi.org/10.1007/s11999-010-1759-9)

---

### **Experimental Design**
We will employ Wizard of Oz experiments to simulate healthcare conversations with **Macaw**, trained on healthcare advice. During these experiments, Macaw will generate responses to simulated healthcare queries related to pain management from users of different gender identities. We will compare the AI-generated responses to existing studies that have documented gender bias in pain management.

Our focus will be on identifying and quantifying gender biases in several key areas:
1. **Tone of language** - Differences in empathy, detail, or clinical tone.
2. **Pain severity acknowledgment** - Disparities in seriousness accorded to pain complaints based on gender.
3. **Treatment recommendations** - Variations in suggestions for men and women (e.g., lifestyle changes for women versus medication for men).
4. **Emotional framing** - Whether women's pain is more often attributed to emotional causes like anxiety or stress compared to men's.

#### **Analytical Methods**
- **Sentiment Analysis**: Assess tone and emotional content of responses.
- **Thematic Analysis**: Identify recurring patterns and themes related to gender bias.
- **Metrics**: Accuracy, consistency, precision, and recall will be used to measure correctness, uniformity, and effectiveness in detecting gender biases.
- **Disparity Index**: Quantifies differences in responses between genders, highlighting gaps in treatment suggestions or tone.

---

### **Expected Outcome**
The project aims to identify gender bias in healthcare conversational agents, particularly related to pain management, and to propose methods for detecting, quantifying, and mitigating these biases. We expect the biases detected in the conversational agent responses to mirror those observed in human healthcare providers, as highlighted in the literature.

---

### **Further Discussion**
What strategies can be employed to ensure fair, accurate, and unbiased healthcare information?

--- 