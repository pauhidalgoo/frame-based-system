import pandas as pd

train_data = pd.read_csv('data/train.csv', header=None, nrows=4078)
print(train_data[2].value_counts())
print()

train_data2 = pd.read_csv('data/train_data_augmented.csv', header=None)
print(train_data2[2].value_counts())
print()

train_sentences = list(train_data[0])
train_labels = list(s.replace('"', '') for s in train_data[2])
train_labels = list(s.replace(' ', '') for s in train_labels)

# Create synthetic data until 50 example per class
 
classes_with_less_than_50_examples = {'aircraft': [], 'quantity': [], 'flight_time': [], 'city': [], 'ground_fare': [], 'distance': [], 'flight+airfare': [],
                                       'airport': [], 'capacity': [], 'flight_no': [], 'meal': [], 'restriction': [], 'airline+flight_no': [],
                                         'ground_service+ground_fare': [], 'airfare+flight_time': [], 'cheapest': [], 'aircraft+flight+flight_no': []}

for sentence, label in zip(train_sentences, train_labels):
    if label in classes_with_less_than_50_examples:
        classes_with_less_than_50_examples[label].append(sentence)

"""for label, sentences in classes_with_less_than_50_examples.items():
    print(label)
    print(len(sentences))
    print(sentences)
    print()"""


# Open the text file with the label-sentence data
with open('baiges/data.txt', 'r') as file:
    lines = file.readlines()

# Initialize variables
label = ""
sentences = []
new_rows = []

# Process each line in the file
for line in lines:
    line = line.strip()

    # If the line is empty, it's a separator between different label groups
    if line == "":
        # Add each sentence to the new rows along with the corresponding label and empty string
        for sentence in sentences:
            new_rows.append([sentence, "", f' "{label}"'])
        
        # Reset the sentences list for the next label group
        sentences = []
        label = None
    
    # If it's the first sentence in the block, it's the label
    elif not label:
        label = line
    else:
        # Add the sentence to the sentence list
        sentences.append(line)



# Convert the new rows to a DataFrame and concatenate it with train_data
new_data = pd.DataFrame(new_rows)
train_data = pd.concat([train_data, new_data], ignore_index=True)

# Save the updated train_data DataFrame
train_data.to_csv('baiges/train_data_augmented.csv', header=False, index=False)