import os
import shutil
import threading #N
#import webbrowser #Kann entfernt werden, sofern die Funktion des automatischen Aufrufs bei jedem Start entfernt werden kann.

#Flask Webserver
from flask import request
from flask import Flask as FlaskServer
from flask import jsonify as JsonResponse
import json

#Vektordatenbank
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

#Importe für das das Symbol in der Taskleiste
from PIL import Image
from PIL import ImageDraw
from pystray import Icon, Menu, MenuItem

#Anforderungen um das Skript inklusive Abhängigkeiten zu starten sind der requirements.txt Datei zu entnehmen.
#Für den Betrieb des Vektordatenservers ist die Installation von der "Visualstudio cpp build tools app" nötig
#Für Visualstudio app install siehe https://visualstudio.microsoft.com/visual-cpp-build-tools/

#Server IP http://127.0.0.1:8000/
#Datenordner für Vektordatenbank mit Unterordnern für übersichtlichere Datenverwaltung
os.makedirs("Vektordatenbank", exist_ok=True)
#Chroma Datenordner
data_folder = "Vektordatenbank/chromadb_files"
os.makedirs(data_folder, exist_ok=True)
#Chroma Datenbank (sqlite3 etc)
database_folder = "Vektordatenbank/chromadb_data"
os.makedirs(database_folder, exist_ok=True)

#Zur protokollierung bereits hochgeladener Dateien. Dopplungen von Dokumenten führen zu mehrfachen Ergebnissen - entwertet die Ergebnisse des Vektorservers.
#Nachträglich zur Problembehebung eingeführt, da Kontextsuchergebnisse sich ständig 1:1 wiederholten.
past_fileuploads_path = "Vektordatenbank/Dokumentenprotokoll.json"

def load_uploaded_files(): #Lädt die Protokoll-Datei für einen Abgleich.
    if os.path.exists(past_fileuploads_path):
        with open(past_fileuploads_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_uploaded_files(): #Speichert die neue vollständige Liste zur Datei.
    with open(past_fileuploads_path, "w", encoding="utf-8") as f:
        json.dump(list(uploaded_files), f, ensure_ascii=False, indent=2)

uploaded_files = load_uploaded_files()

#Flask - API Interaktion (Entwicklungsserver - Sollte ersetzt werden laut Konsole durch einen WSGI Produktionsserver)
server = FlaskServer(__name__)


#Globale Variable für die Vektordatenbank
vector_db = None
#Aus langchain.embeddings Bibliothek Modell zur semantischen Erfassung - vortrainierter Transformer
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Bitte ändern, falls in der Zukunft die Datenbank jedes Mal neu aufgebaut werden soll.
def initialize_database(reset_db=False):
    global vector_db

    if reset_db:
        #Zurücksetzen der Vektordatenbank.
        shutil.rmtree(database_folder, ignore_errors=True)
        #Ordner erneut erstellen.
        os.makedirs(database_folder, exist_ok=True)
        print("Vektordatenbank wurde zurückgesetzt.")

    if vector_db is None:
        #Datenbank erzeugen/laden
        vector_db = Chroma(persist_directory=database_folder, embedding_function=embedding_model)
        print("Chroma Vektordatenbank geladen.")

#Initialisiere Vektordatenbank beim Programmstart.
#Setze zu True,falls in der Zukunft die Datenbank per Default jedes Mal neu aufgebaut werden soll. (nur falls kein Parameter mit dem Kommando verwendet wird.)
initialize_database(reset_db=False)

#API-Schnittstelle zum Hochladen auf diesen Server zur Verarbeitung
#Hinweis: die JsonResponse Antworten könnten besser in dem GUI Programm implementiert werden.
@server.route("/upload", methods=["POST"])
def file_upload():
    global uploaded_files
    #Dateianhang und Dateinamen Prüfung
    if "file" not in request.files:
        print("Anfrage enthält keine Datei. Error 401")
        return JsonResponse({"error": "Anfrage enthält keine Datei."}), 401

    file = request.files["file"]
    if file.filename == "":
        print("Datei hat keinen Namen. Error 402")
        return JsonResponse({"error": "Datei hat keinen Namen."}), 402

    file_path = os.path.join(data_folder, file.filename)

    #Überprüfe, ob die Datei bereits hochgeladen wurde
    if file.filename in uploaded_files:
        print("Datei wurde bereits hochgeladen. Error 403")
        return JsonResponse({"error": "Datei wurde bereits hochgeladen."}), 403

    file.save(file_path)
    print(f"File uploaded successfully to {file_path}.")

    #Dokumente einlesen und in die Vektordatenbank implementieren/einbetten
    #Bisher unterstützt sind nur txt, pdf und docx Formate
    try:
        loader = None
        if file.filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Skipping unsupported file type: {file.filename}")
            print("Dateityp wird nicht unterstützt. Error 400")
            return JsonResponse({"error": "Dateityp wird nicht unterstützt."}), 400

        if loader:
            docs = loader.load()

            #Text wird in Chunks aufgeteilt. Zusätzliche Parameter wurden gewählt, um die Chunks zu vergrößern und so einen größeren zusammenhängenden Kontext bei Anfragen wiederzugeben.
            #overlap gibt die kontextuelle Überschneidung vor. size legt die größe der Chunks fest.
            #Bedarf feinjustierung und wurde sehr hoch angesetzt, da kleine Kontexte zusammenhangslos als Antwort wiedergegeben wurden, sodass das LLM keine sinnvolle Antwort gegeben hat.
            #Achtung: bei weitergabe an OpenAI mit großen Dokumenten & großen Chunks erhöht sich massive die größe der Verbrauchten Input Tokens und erhöht dadurch kosten.
            #Bei Verwendung besonders kostenintensiver OpenAI Modelle wie "o1" sind die resultierenden Kosten nicht absehbar. Kontext: eine kleine "gpt4o" Anfrage kostet ca 4 cent
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,    #chunk_size ist die Größe des Dokuments/Kontextes
                chunk_overlap=110,  #Kontextuelle überschneidung
                separators=["\n\n", "\n", " ", ""])

            split_documents = text_splitter.split_documents(docs)

            #Chunks in die Vektordatenbank hinzufügen
            vector_db.add_documents([Document(page_content=doc.page_content) for doc in split_documents])
            print("Dokument " + file.filename +" hinzugefügt.")

            #Dateinamen in der Liste von bereits hochgeladenen Dateien ablegen.
            uploaded_files.add(file.filename)
            save_uploaded_files()
        else:
            print("Fehler - Es wurde kein Embeddings hinzugefügt.")

    except Exception as e:
        print(f"Fehler beim Laden/verarbeiten: {file.filename}: {e}")
        return JsonResponse({"error": str(e)}), 500

    return JsonResponse({"message": "Datei wurde erfolgreich hochgeladen! Die Verarbeitung des Dateiinhaltes kann etwas Zeit in Anspruch nehmen."})

#Webserver Adresse zur Stellung von Anfragen.
@server.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json #Eingehnde API-Anfrage
        if not data or "question" not in data:
            #Fehlende Stelle in der API-Anfrage.
            return JsonResponse({"error": "Json Anfrage enthält keinen 'question' key"}), 400

        #Im folgenden Abschnitt Zeile 170 max_results wird die maximale Anzahl relevanter Dokumente festgelegt
        #Die Zahl ist sehr hoch gewählt, da bei kleiner max_results Anzahl trotz vielzähliger Anhänge nur sehr wenige
        #Resultate an die LLM weitergegeben wurden. Dies ist definitiv ein Parameter, der bei der Verwendung von OpenAI
        #Servern Verwendung abhängig vom Modell kostenintensiv sein könnte.
        question = data["question"]
        max_results = data.get("max_results", 3)  #Anzahl ändern, um die maximale Anzahl relevanter Dokumente zurückzugeben. 3 ist ein default Wert falls nichts anderes angegeben wurde.

        #Frage wird an die Vektordatenbank "gestellt".
        retriever = vector_db.as_retriever(search_kwargs={"k": max_results}) #search kwargs wird im folgenden "max_results" für ein besseres Verständnis umbenannt.
        #Alternativ ohne max_results festzulegen
        #retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        print(f"Anzahl der gefundenen Dokumente: {len(docs)} für die Frage: '{question}'") #Feedback darüber wie viele Dokumente gefunden wurden.

        #Antwort an den Client PC/an die GUI, um Kontext bereitzustellen.
        return JsonResponse({
            "documents": [{"content": doc.page_content} for doc in docs]
        })

    except Exception as e:
        print("Ein Fehler ist aufgetreten:", e)
        return JsonResponse({"error": f"Ein Fehler ist aufgetreten: {str(e)}"}), 500
#Informationsseite/Startseite. Zum Testen der Verbindung sowie der kleinen Dokumentation über die möglichen API Schnittstellen
@server.route("/", methods=["GET"])
def index():
    return "Vektordatenbank ist aktiv. Dateien hochladen unter /upload. Kontextanfragen stellen under /ask."

def start_server():
    port = 8000
    url = f"http://127.0.0.1:{port}"


    #Öffnet den Webbrowser bei Start des Servers. Import Kommentierung Rückgängig machen, um diese Funktion wieder zu aktivieren.
    #threading.Timer(1, lambda: webbrowser.open(url)).start()

    print(f"Server läuft unter {url}")
    server.run(port=port, debug=False)

#########
#System Tray Icon, kann ebenfalls mit existierendem Bild umgesetzt werden.
#Vorschlag der Nutzung des RUB Logos, dabei ist mir unklar, ob ich über die Verwendungsrechte verfüge.
#########

#Einfaches System Tray Icon erstellen
def create_icon():
    width, height = 64, 64
    icon = Image.new('RGB', (width, height), "white")
    draw = ImageDraw.Draw(icon)
    draw.rectangle((0, 0, width, height), fill="blue")
    draw.ellipse((16, 16, width - 16, height - 16), fill="white")
    return icon

#Rechtsklick Option zum Beenden des Programms ohne im IDE oder im Taskmanager greifen zu müssen.
def stop_server(icon, item):
    icon.stop()
    os._exit(0)  #Schließt alle Aufgaben & Server

#Startet das Taskleistensymbol zum einfacheren Beenden.
def start_tray_icon():
    menu = Menu(MenuItem("Beenden", stop_server))
    icon = Icon("Vektordatenbank Server", create_icon(), menu=menu)
    icon.run()

#Starte Hauptprogramm mit Threading, da Sys Tray Icon + Server nicht gleichzeitig betrieben werden können - Führt zu Systemeinfrieren und Fehlern.
if __name__ == "__main__":
    tray_thread = threading.Thread(target=start_tray_icon, daemon=True)
    tray_thread.start()
    start_server()
