# LLM-RAG-Python-GUI
LLM Nutzeroberfläche mit RAG Kompabilität und Vektorserver

Anforderungen zur Installation der Abhängigkeiten: 
```
#LLM GUI
python-dotenv
requests
python-docx
#Vektordatenbank
Flask
langchain
chromadb
Pillow
pystray
```
Für den Betrieb des Vektordatenservers ist die Installation von der "Visualstudio cpp build tools app" nötig
Für Visualstudio app install siehe https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Empfohlener Aufbau der .env
```
#OpenAI
OPENAI_KEY=
OPENAI_MODELS=gpt-4o-mini,gpt-4o,gpt-4,gpt-4-turbo,o1-mini,o1,o1-pro
OPENAI_API_URL=https://api.openai.com/v1/chat/completions

#Lokaler LLM Server
LOCAL_IP = 127.0.0.1
LLM_MODELS=llama3.2:3b

#Vektorserver
VECTORDB_URL=http://127.0.0.1:8000
VECTORDB_API=NONE
VECTORDB_PORT=11434
```
