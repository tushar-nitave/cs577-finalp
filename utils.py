"""
This is a utiliy module which consists of various parameters required in main method.
"""
CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.,;:><{}'|~`[]=?/!@#$%^&*()_\"\xa3\xc3\x89\xe9\xa9\x91\xd1\xc9\xb4` \\\n"

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 128, 32

# Network parameters
batch_size = 128
val_batch_size = 16

downsample_factor = 4
max_text_len = 10

epochs = 10
