#Python 3.7.9
#OpenCV 4
import cv2
import numpy as np
import os

def manual_threshold_imagem(img,type_thres = 1):
    result = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            if type_thres == 1:
                if img[i,j]>=50:
                    result[i,j] = 255
                else:
                    result[i,j] = 0
            elif type_thres == 2:
                if img[i,j]>=50:
                    result[i,j] = 0
                else:
                    result[i,j] = 255

    return result.astype('uint8')

def sharpen_img(img):  # Melhorando detalhes
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    result = cv2.filter2D(img,-1,kernel)
    return result

def checacores(img):
    if img[:,:,0] > img[:,:,1] and img[:,:,0] > img[:,:,2]:
        return "azul"
    elif img[:,:,0] < img[:,:,1] and img[:,:,1] > img[:,:,2]:
        return "verde"
    elif img[:,:,2] > img[:,:,0] and img[:,:,1] < img[:,:,2]:
        return "vermelho"
    
def contapecas(img,cor):
    raw_img = img.copy()

    img = img[7:img.shape[0]-7,7:img.shape[1]-7,:] #Cortando bordas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray,3) # Homogeinizando comportamento dos pixels
    img = sharpen_img(img) # Aumentando detalhes


    thr = manual_threshold_imagem(gray,1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.erode(thr,kernel,iterations = 1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 200
    max_area = 550
    pecas_vermelhas = []
    pecas_verdes = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)          
            ROI = raw_img[y:y+h, x:x+h]
            if(checacores(ROI[11:12,11:12,:])=="verde"): #Utilizando apenas o pixel central da area de interesse pra fazer a checagem
                pecas_verdes.append(ROI)
            elif(checacores(ROI[11:12,11:12,:])=="vermelho"):
                pecas_vermelhas.append(ROI)

    if (cor == "vermelho"):
        return "A imagem possui "+str(len(pecas_vermelhas))+" peças vermelhas"
    elif (cor == "verde"):
        return "A imagem possui "+str(len(pecas_verdes))+" peças verdes"
    

def removepecas(img):
    raw_img = img.copy()
    img = img[7:img.shape[0]-7,17:img.shape[1]-27,:] #Cortando bordas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray,3)
    img = sharpen_img(img)


    thr = manual_threshold_imagem(gray,1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.erode(thr,kernel,iterations = 1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 200
    max_area = 550

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)          
            raw_img[2+y:y+h+10, 13+x:x+h+21] = 0 # Corrigindo o deslocamento causado pelo corte inicial e aplicando a nova cor
    return raw_img

def segmentapecas(img,mascara):
    mascara = mascara[7:mascara.shape[0]-7,65:mascara.shape[1]-67,:] #Cortando bordas
    raw_img = img[7:img.shape[0]-7,65:img.shape[1]-67,:]
    gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
    mascara = cv2.medianBlur(gray,3)
    mascara = sharpen_img(mascara)
    listadequadros = []


    thr_brancas = manual_threshold_imagem(mascara,1)
    thr_pretas = manual_threshold_imagem(mascara,2)

    thr = [thr_brancas,thr_pretas]
    for img_preprocessada in thr:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.erode(img_preprocessada,kernel,iterations = 1)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        min_area = 200
        max_area = 1550
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)          
                ROI = raw_img[y:y+h, x:x+h]
                listadequadros.append(ROI)

    return listadequadros

def Test1():
    img = cv2.imread("ImagemA.jpeg")
    result = contapecas(img,"verde") #vermelho
    print(result)

def Test2():
    img = cv2.imread("ImagemB.jpeg")
    tabuleirofiltrado = removepecas(img)
    listadepecas = segmentapecas(img,tabuleirofiltrado)
    listadepecas.reverse() #Colocando a lista de segmentos na sequencia correta comparado ao tabuleiro na imagem de entrada
    print("Quantidade de quadros: "+ str(len(listadepecas)))
    b=0
    a=0

    if not os.path.exists("Segmentados"):
        os.makedirs("Segmentados")

    
    mid = int(len(listadepecas)/2)

    imgsA = listadepecas[0:mid]
    imgsB = listadepecas[mid:100]
    for i in range(0,50):

        cv2.imwrite("Segmentados/"+str(a)+str(b)+'.jpg',imgsB[i])
        b = b+1
        cv2.imwrite("Segmentados/"+str(a)+str(b)+'.jpg',imgsA[i])
        b = b+1
        if b == 10:
            a = a+1
            b = 0
        


    while(1):
        cv2.imshow("Pequeno segmento do tabuleiro",listadepecas[0])
        k = cv2.waitKey(33)
        if k==27:    #ESC pra sair
            break
            cv2.destroyAllWindows()
        elif k==-1:  
            continue


def Test3():
    img = cv2.imread("ImagemC.jpeg")
    test = removepecas(img)

    while(1):
        cv2.imshow("Tabuleiro Filtrado",test)
        k = cv2.waitKey(33)
        if k==27:   #ESC pra sair
            break
            cv2.destroyAllWindows()
        elif k==-1:  
            continue




#Test1()
Test2()
#Test3()

