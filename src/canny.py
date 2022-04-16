import matplotlib.pyplot as plt
import numpy as np
import cv2


def gray_scale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gauss_kernel(size, sigma):
    kernel = np.zeros((size, size))
    val = 1 / (2 * np.pi * np.power(sigma, 2))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = val * np.exp(
                (np.square(i - (size // 2 + 1)) + (np.square(j - (size // 2 + 1)))) / (- 2 * np.square(sigma)))
    return kernel


def convolve(img, kernel):
    if len(kernel) % 2 != 0:
        pad = np.pad(img, ((len(kernel) - 1) // 2, (len(kernel) - 1) // 2))
    else:
        pad = np.pad(img, (len(kernel) - 1, len(kernel) - 1))
    new_image = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_image[i][j] = (pad[i:i + len(kernel), j:j + len(kernel)] * kernel).sum()
    return new_image


def sobel(img):
    fil = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    hor_filter = convolve(img, fil)
    ver_filter = convolve(img, np.transpose(fil))
    gradient_magnitude = np.sqrt(np.square(hor_filter) + np.square(ver_filter))
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    theta = np.arctan2(ver_filter, hor_filter)
    return gradient_magnitude, theta


def edge_compression(img, theta):
    compressed = np.zeros(img.shape)
    theta = np.rad2deg(theta)
    theta[np.where(theta[:, :] < 0)] += 180
    for i in range(1, img.shape[0] - 2):
        for j in range(1, img.shape[1] - 2):
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                compare = max(img[i, j - 1], img[i, j], img[i, j + 1])
            elif 22.5 <= theta[i, j] < 67.5:
                compare = max(img[i - 1, j - 1], img[i, j], img[i + 1, j + 1])
            elif 67.5 <= theta[i, j] < 112.5:
                compare = max(img[i - 1, j], img[i, j], img[i + 1, j])
            else:
                compare = max(img[i + 1, j - 1], img[i, j], img[i - 1, j + 1])
            if img[i, j] == compare:
                compressed[i, j] = img[i, j]
    return compressed


def threshold(img, weak, strong):
    high = img[:, :].max() * 0.09
    low = high * 0.05
    output = np.zeros(img.shape)
    weak_i, weak_j = np.where((low < img[:, :]) & (img[:, :] <= high))
    strong_i, strong_j = np.where(img[:, :] > high)
    output[weak_i, weak_j] = weak
    output[strong_i, strong_j] = strong
    return output


def hysteresis(img, weak, strong):
    pos_x, pos_y = np.where(img[:, :] == weak)
    for i in range(len(pos_x)):
        if strong in img[pos_x[i] - 1:pos_x[i] + 1, pos_y[i] - 1:pos_y[i] + 1]:
            img[pos_x[i], pos_y[i]] = strong
        else:
            img[pos_x[i], pos_y[i]] = 0
    return img


def canny_edge(img):
    weak = 45
    strong = 255
    gauss_fil = gauss_kernel(5, 1.6)
    blur_img = convolve(img, gauss_fil)
    gradient, angle = sobel(blur_img)
    edge_com = edge_compression(gradient, angle)
    db_th = threshold(edge_com, weak, strong)
    final = hysteresis(db_th, weak, strong)
    plt.imshow(final, cmap='gray')
    plt.savefig('./images/Lizard_Edges.jpg')
    plt.show()


def canny_cv(img):
    cv2.Canny(img, 15, 25, apertureSize=5, L2gradient=True)


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread('./images/Lizard.jpg'), cv2.COLOR_BGR2RGB)
    canny_cv(image)
    image = gray_scale(image)
    canny_edge(image)
