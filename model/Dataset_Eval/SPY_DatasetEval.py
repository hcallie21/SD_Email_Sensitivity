from datasets import load_dataset

# Load the SPY dataset with dynamic entity generation
dataset = load_dataset("spy.py", trust_remote_code=True, split="medical_consultations", faker_random_seed=0)
print(dataset)