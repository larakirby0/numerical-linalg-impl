import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from time import time

IMG_PATH = "/Users/larakirby/Desktop/Screenshot 2025-08-04 at 9.07.31 AM.png"

def preprocess_image(IMG_PATH) -> np.array:
    # 1) Load and preprocess the image
    img = Image.open(IMG_PATH).convert('L') # Convert to grayscale
    img = img.resize((img.width // 2, img.height // 2)) # Optional: downsample
    img_array = np.array(img) / 1023.0 # Scale to [0,1]
    print("==========")
    print(img_array)
    print("----------")
    dims = img_array.shape
    print("Dimensions:", dims)
    return img_array

def compress_image(img_array, ranks=[320, 160, 40, 10]):
    """
    Load an image, convert it to grayscale, perform SVD, and
    display the original plus compressed versions in separate windows.
    """
    # Display the original image in its own window
    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title('Black and White')
    plt.axis('off')
    # Show (non-blocking) so that subsequent windows can appear
    plt.show(block=False)
    
    start = time()
    U, S, Vt = np.linalg.svd(img_array, full_matrices=False)
    # For each rank k, reconstruct and display the compressed image in its own window
    for k in ranks:
    # Keep only the first k singular values
        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        Vtk = Vt[:k, :]
        compressed_img = Uk @ Sk @ Vtk
        plt.figure()
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f'Rank {k} SVD')
        plt.axis('off')
        plt.show(block=False)
    svd_time = time() - start
    print(f"SVD in {svd_time}")
    # Finally, keep all figures open until you manually close them
    plt.show()


def compress_image_QR(img_array, ranks=[160, 40, 10]):
    """
    Load an image, convert it to grayscale, perform QR, and
    display the original plus compressed versions in separate windows.
    """
    # 2) Compute the SVD
    start = time()
    Q, R = np.linalg.qr(img_array)
    qr_time = time() - start
    print(f"QR in {qr_time}")
    # 3) Display the original image in its own window
    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title('Black and White')
    plt.axis('off')
    # Show (non-blocking) so that subsequent windows can appear
    plt.show(block=False)
    # 4) For each rank k, reconstruct and display the compressed image in its own window
    for k in ranks:
        Qk = Q[:, :k]
        Rk = R[:k, :]
        compressed_img = Qk @ Rk
        plt.figure()
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f'Rank {k} QR')
        plt.axis('off')
        plt.show(block=False)
    
    # Finally, keep all figures open until you manually close them
    plt.show()

def compress_image_NMF(img_array, ranks=[160, 40, 10]):
    """
    Load an image, convert it to grayscale, perform SVD, and
    display the original plus compressed versions in separate windows.
    """
    # Display the original image in its own window
    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title('Black and White')
    plt.axis('off')
    # Show (non-blocking) so that subsequent windows can appear
    plt.show(block=False)
    # For each rank k, reconstruct and display the compressed image in its own window
    start = time()
    for k in ranks:
        nmf = NMF(n_components=k, max_iter=10000)
        N = nmf.fit_transform(img_array)
        H = nmf.components_
        compressed_img = N @ H
        plt.figure()
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f'Rank {k} NMF')
        plt.axis('off')
        plt.show(block=False)
    nmf_time = time() - start
    print(f"NMF in {nmf_time}")
    # Finally, keep all figures open until you manually close them
    plt.show()

def image_with_noise(img_array) -> np.array:
    dims = img_array.shape
    noise = np.random.rand(dims[0], dims[1]) / 10
    with_noise = img_array + noise

    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title('Image without noise')
    plt.axis("off")
    plt.show(block=False)
    plt.figure()
    plt.imshow(with_noise, cmap='gray')
    plt.title('Image with noise')
    plt.axis('off')
    plt.show()
    return with_noise


def compress_image_CUR(img_array, ranks=[160, 40, 10]):
    # 3) Display the original image in its own window
    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title('Black and White')
    plt.axis('off')
    # Show (non-blocking) so that subsequent windows can appear
    plt.show(block=False)
    dims = img_array.shape

    number_of_rows = dims[0]
    number_of_columns = dims[1]
    
    start = time()
    for k in ranks:
        random_column_choices = np.random.choice(number_of_columns, size=k, replace=False)
        random_row_choices = np.random.choice(number_of_rows, size=k, replace=False)
        C = img_array[:, random_column_choices]
        R = img_array[random_row_choices, :]

        C_plus = np.linalg.pinv(C)
        R_plus = np.linalg.pinv(R)

        U = C_plus @ img_array @ R_plus

        compressed_img = C @ U @ R
        plt.figure()
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f'Rank {k} CUR')
        plt.axis('off')
        plt.show(block=False)

    cur_time = time() - start
    print(f"CUR in {cur_time}")
    plt.show()

if __name__ == "__main__":

    image_array = preprocess_image(IMG_PATH)
    svd = compress_image(image_array, ranks=[320, 160, 40, 10])
    compress_image_QR(image_array, ranks=[320, 160, 40, 10])
    # compress_image_NMF(image_array, ranks=[320, 160, 40, 10])
    compress_image_CUR(image_array,ranks=[320, 160, 40, 10] )

    image_array_with_noise = image_with_noise(image_array)
    compress_image(image_array_with_noise, ranks=[320, 160, 40, 10])
    compress_image_QR(image_array_with_noise, ranks=[320, 160, 40, 10])
    # compress_image_NMF(image_array_with_noise, ranks=[320, 160, 40, 10])
    compress_image_CUR(image_array_with_noise,ranks=[320, 160, 40, 10] )