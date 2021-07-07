import cv2 
import time 

#CORES DAS CLASSES
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

#CARREGA AS CLASSES
class_names =[]
with open("coco.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#CAPITURANDO VIDEO    
#caso deseje usar a camera é so alterar apagar o q esta entre parenteses e colocar 0
#case queira testar com outros videos so colocar na mesma pasta do projeto, colocar o nome dele entre aspas como feito abaixo
cap=cv2.VideoCapture("hehe.mp4")

#carregando pessos da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

#setando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

#lendo os frames do video

while True:
    #capiturando frame
    _, frame = cap.read()

    # comeco da contagem dos ms
    start = time.time()

    #deteccao
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    #fim da contagem dos ms
    end = time.time()
     
    #percorrer todas as deteccoes
    for(classid, score,box) in zip(classes, scores, boxes):

        #gerando uma cor para a classe
        color =COLORS[int(classid)%len(COLORS)]

        #PEGANDO O NOME DA CLASSE PELO ID E SEU SCORE DE ACURACIA
        label = f"{class_names[classid[0]]} : {score}"
        
        #desenhando a box da deteccao
        cv2.rectangle(frame,box,color,2)

        #escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
    #calcuculando o tempo q levou para detectar
    fps_label =f"FPS: {round((1.0/(end-start)),2)}"    

    #escrevendo o FPS na imagem
    cv2.putText(frame,fps_label,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
    cv2.putText(frame,fps_label,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    #mostrando imagem
    cv2.imshow("detections",frame)

    #espera da resposta
    if cv2.waitKey(1) == 27:
        break
#liberacao da macera e destroi as janelas 
cap.release()
cv2.destroyAllWindows()    
