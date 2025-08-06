import numpy as np

# Load both .npz files
train_data = np.load('./my_beatles_data/test_data.npz', allow_pickle=True)
test_data = np.load('./my_beatles_data/augmented_data.npz', allow_pickle=True)

print("beatles keys:", train_data.files)
print("beatles keys:", test_data.files)


train_data = np.load('./my_harmonix_data/test_data.npz', allow_pickle=True)
test_data = np.load('./my_harmonix_data/train_data.npz', allow_pickle=True)

print("harmonix keys:", train_data.files)
print("harmonix keys:", test_data.files)

train_data = np.load('./my_salami_data/test_data.npz', allow_pickle=True)
test_data = np.load('./my_salami_data/train_data.npz', allow_pickle=True)

print("salami keys:", train_data.files)
print("salami keys:", test_data.files)

# train_data = np.load('./beatles_data/augmented_data.npz', allow_pickle=True)
# test_data1 = np.load('./salami_data/train_data.npz', allow_pickle=True)

# Check available keys
# print("beatles keys:", train_data.files)
# print("beatles keys:", test_data.files)
# print("beatles train keys:", test_data2.files)
# print("salami keys:", test_data1.files)

# # Create a new dictionary to hold combined data
# combined_data = {}

# # Combine data for each key (assuming same keys in both files)
# for key in train_data.files:
#     combined_data[key] = np.concatenate((train_data[key], test_data[key]), axis=0)

# # Save to new combined file
# np.savez('./my_beatles_data/train_data.npz', **combined_data)

# print("Combined data saved as train_data.npz")
