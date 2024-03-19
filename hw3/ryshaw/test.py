import hashlib
import random
import string


salt = "e89dd8d78728"

answer = "578e96cad50e5023eef9c57081286949"


#iterate through all words in words.txt
# with open("words.txt") as f:
#     for line in f:
#         word = line.strip()
#         temp = hashlib.md5(bytes(word+salt,encoding="utf-8")).hexdigest()
#         if temp == answer:
#             print(word)
#             break

#print all 7 character words in words.txt where the lowercase letters are given a value 1-26 and the letters sum to 42
# with open("words.txt") as f:
#     for line in f:
#         word = line.strip()
#         if len(word) == 7:
#             sum = 0
#             for letter in word:
#                 sum += int(ord(letter)) - 96
#             if sum == 42 and word[0] != 'h' and 'l' not in word:
#                 print(word)


#print all 7 letter words that include 'w' 'r' 'o' 'u' 'g' 'h' and 't' in words.txt 
with open("words.txt") as f:
    for line in f:
        word = line.strip()
        if len(word) == 7:
            if 'w' in word and 'r' in word and 'o' in word and 'u' in word and 'g' in word and 'h' in word and 't' in word:
                print(word)

