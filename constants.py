from dotenv import load_dotenv
import os

load_dotenv("../.env")

path = os.get("IMG_PATH")

ca = [4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # fmt: skip
ci = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8] # fmt: skip
cj = [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 1, 3, 6, 2, 4, 7, 3, 5, 8, 1, 4, 6, 2, 5, 7, 3, 6, 8, 4, 7] # fmt: skip
b = [6, 7, 7, 7, 7, 7, 7, 6]
epsilon = 10**-5
omega = 1.2
# arbitrary initial guess, since these methods should all converge for any choice of x_0
initial_guess = [0.9, 0.9, 0.9, 0.9, 1, 1, 1, 1]