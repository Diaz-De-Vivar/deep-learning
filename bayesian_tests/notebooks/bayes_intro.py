import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette('husl')

# Define probabilities
P_disease = 0.01  # Prior probability of having the disease
P_positive_given_disease = 0.95  # Probability of positive test given disease
P_negative_given_no_disease = 0.95  # Probability of negative test given no disease

# Calculate P(positive test)
P_positive = (P_positive_given_disease * P_disease + 
             (1 - P_negative_given_no_disease) * (1 - P_disease))

# Calculate posterior probability using Bayes' Theorem
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"Probability of having the disease given a positive test: {P_disease_given_positive:.2%}")

# Create visualization
plt.figure(figsize=(10, 6))

# Plot prior and posterior
plt.bar(['Prior', 'Posterior'], 
        [P_disease, P_disease_given_positive],
        color=['lightblue', 'lightgreen'])

plt.title('Prior vs Posterior Probability of Having the Disease')
plt.ylabel('Probability')
plt.ylim(0, 1)

# Add value labels on top of bars
for i, v in enumerate([P_disease, P_disease_given_positive]):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center')

plt.show() 