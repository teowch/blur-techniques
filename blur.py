import sys
import timeit
import numpy as np
import cv2

INPUT_IMAGE = "tree.jpg"


def boxBlur(img, fullw):
    w = fullw // 2
    img_return = np.array
    img_return = img.reshape((img.shape[0], img.shape[1], 3))

    # For each pixel...
    for y in range(w, len(img) - w):
        for x in range(w, len(img[0]) - w):
            soma = 0
            for i in range(-w, w + 1):
                for j in range(-w, w + 1):
                    # soma cada pixel da janela
                    soma += img[y + i, x + j]
            media = soma / (fullw**2)
            img_return[y, x] = media

    return img_return


def separableBoxBlur(img, fullW):
    halfW = fullW // 2

    img_horizontal = np.copy(img)

    for y in range(halfW, len(img) - halfW):
        for x in range(halfW, len(img[0]) - halfW):
            # para cada pixel
            soma = 0
            for i in range(-halfW, halfW + 1):
                soma += img[y, x + i]
            media = soma / fullW
            img_horizontal[y, x] = media

    img_return = np.copy(img_horizontal)

    for y in range(halfW, len(img) - halfW):
        for x in range(halfW, len(img[0]) - halfW):
            soma = 0
            for i in range(-halfW, halfW + 1):
                soma += img_horizontal[y + i, x]
            img_return[y, x] = soma / fullW

    return img_return


def integralBlur(img, fullW):
    halfW = fullW // 2
    img_return = np.copy(img)
    img_integral = np.array
    img_integral = img.reshape((img.shape[0], img.shape[1], 3))
    # criação da imagem integral --------------------------------
    # para cada linha y
    for y in range(0, len(img)):
        # primeiro px da linha é igual o original
        img_integral[y, 0] = img[y, 0]
        # para cada coluna fora a primeira
        for x in range(1, len(img[0])):
            # pixel é px original mais integral à esquerda
            img_integral[y, x] = img[y, x] + img_integral[y, x - 1]

    # para cada linha y fora a primeira
    for y in range(1, len(img)):
        # para cada coluna x
        for x in range(0, len(img[0])):
            # pixel é igual a ele mais pixel de cima
            img_integral[y, x] += img_integral[y - 1, x]
    # print(img_integral)

    # Obtenção da média ------------------------------------------
    # para cada pixel...
    for y in range(0, len(img)):
        for x in range(0, len(img[0])):
            # para cada pixel, janela:
            soma = 0
            xSoma = x + halfW
            ySoma = y + halfW
            # soma += img_integral[y+halfW, x+halfW]
            winW = fullW
            winH = fullW

            # px na borda direita
            if x > (len(img[0]) - 1 - halfW):
                xSoma = len(img[0]) - 1
                winW = halfW + len(img[0]) - 1 - x
            # px na borda inferior
            if y > (len(img) - 1 - halfW):
                ySoma = len(img) - 1
                winH = halfW + len(img) - 1 - y

            soma += img_integral[ySoma, xSoma]

            if y > halfW:
                # subtrai px superior direito de fora da janela
                soma -= img_integral[y - halfW - 1, xSoma]
            else:
                # ajusta largura da janela: mais baixa que fullW
                winH = y + halfW

            if x > halfW:
                # subtrai px inferior esquerdo de fora da janela
                soma -= img_integral[ySoma, x - halfW - 1]
            else:
                # ajusta largura da janela: mais estreita que fullW
                winW = x + halfW

            if x > halfW and y > halfW:
                # soma px superior esquerdo de fora da janela
                soma += img_integral[y - halfW - 1, x - halfW - 1]

            """ 
            # nos pixels que não são bordas:
            soma = img_integral[y+halfW, x+halfW] - img_integral[y-halfW-1, x+halfW] - img_integral[y+halfW, x-halfW-1] + img_integral[y-halfW-1, x-halfW-1]
            media = soma / (fullW**2)
            img_return[y, x] = media
            """

            media = soma / (winH * winW)
            img_return[y, x] = media

    return img_return


def main():
    ingenuo = True
    separavel = True
    integral = True

    # Leitura do arquivo-----------------------------------
    # img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)

    if img is None:
        print("Cannot open image")
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], 3))
    img = img.astype(np.float32)
    img /= 255

    # Algoritmos
    if ingenuo:
        start_time = timeit.default_timer()
        img_output = boxBlur(img, 9)
        cv2.imwrite("04 - blurIngenuo.png", img_output * 255)
        print("Tempo ingênuo: %f" % (timeit.default_timer() - start_time))
        cv2.imshow("Ingenuo", img_output)
    if separavel:
        start_time = timeit.default_timer()
        img_output = separableBoxBlur(img, 15)
        cv2.imwrite("04 - blurSeparavel.png", img_output * 255)
        print("Tempo separável: %f" % (timeit.default_timer() - start_time))
        cv2.imshow("Separavel", img_output)
    if integral:
        start_time = timeit.default_timer()
        img_output = integralBlur(img, 55)
        cv2.imwrite("04 - blurIntegral.png", img_output * 255)
        print("Tempo integral: %f" % (timeit.default_timer() - start_time))
        cv2.imshow("Integral", img_output)

    # cv2.imshow('Output', img_output)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
