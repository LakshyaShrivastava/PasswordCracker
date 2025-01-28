import hashlib
from algorithms import brute_force

# brute_force_test
def brute_force_test():
    target_hash = hashlib.sha256(b"abc123").hexdigest()
    cracked_password = brute_force(target_hash, max_length=6)
    if cracked_password:
        print(f"Password cracked: {cracked_password}")
    else:
        print("Password not found.")

if __name__ == '__main__':
    brute_force_test()
