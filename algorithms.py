import hashlib
import itertools
import string

DEBUG_PRINT = False

def brute_force(hash_to_crack, max_length):
    # characters to include in brute force
    characters = string.ascii_lowercase + string.digits

    for length in range(1, max_length + 1):
        # generate all possible combinations
        for guess in itertools.product(characters, repeat=length):
            guess_word = ''.join(guess)
            if DEBUG_PRINT:
                print("Guessing " + guess_word)
            hashed_guess = hashlib.sha256(guess_word.encode()).hexdigest()
            if hashed_guess == hash_to_crack:
                return guess_word
    return None
