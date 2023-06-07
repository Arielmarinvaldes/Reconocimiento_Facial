from modulos import face_recognizer as fr

def reconocimiento():
    res = input("activar o desactivar: ")
    # Inicializamos la camara
    if res == 'activar':
        t = tr.Thread(target=fr.face_rec, args=(0,))
        t.start()
        print("Activando reconocimiento")
    elif 'desactivar':
        print("Desactivando reconocimiento")
        fr.face_rec(1)