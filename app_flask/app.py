import os
from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import joblib 
import time
import threading
import webbrowser
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assumiamo che 'paziente_test2.csv' sia nella stessa cartella di app.py
SIGNAL_FILE = os.path.join(BASE_DIR, 'test_patient.csv')

# Carichiamo il modello in memoria (simulato tramite try-except per farti testare il codice anche senza modello)
try:
    modello = joblib.load(os.path.join(BASE_DIR, 'modello_epilessia.pkl'))
except Exception as e:
    print(f"Errore caricamento modello: {str(e)}")
    modello = None
    print("Avviso: Modello non trovato. Assicurati di avere 'modello_epilessia.pkl' per le previsioni reali.")

# --- ROTTE PAZIENTE ---
@app.route('/', methods=['GET'])
def home():
    # Di default mostriamo la dashboard e NON mostriamo il dettaglio analisi
    return render_template('index.html', previsione=None, errore=None, active_tab='dashboard', show_detail=False)

@app.route('/analisi', methods=['GET'])
def lista_analisi():
    # Forza l'apertura del tab analisi in modalità lista
    return render_template('index.html', previsione=None, errore=None, active_tab='eeg', show_detail=False)

@app.route('/analisi/dettaglio', methods=['GET'])
def dettaglio_analisi():
    # Forza l'apertura del tab analisi in modalità dettaglio (il referto)
    return render_template('index.html', previsione=None, errore=None, active_tab='eeg', show_detail=True)

# NUOVA ROTTA per servire i dati del segnale
@app.route('/get_signal', methods=['GET'])
def get_signal():
    if not os.path.exists(SIGNAL_FILE):
        return jsonify({'errore': 'File segnale non trovato.'}), 404
    
    try:
        # Leggiamo il CSV con pandas e appiattiamo i dati in una lista
        dati = pd.read_csv(SIGNAL_FILE, header=None)
        signal_data = dati.T.values.flatten().tolist()
        return jsonify(signal_data)
    except Exception as e:
        return jsonify({'errore': f"Errore lettura segnale: {str(e)}"}), 500

# NUOVA ROTTA per il download del file
@app.route('/download_signal', methods=['GET'])
def download_signal():
    if not os.path.exists(SIGNAL_FILE):
        return "File segnale non trovato.", 404
    return send_file(SIGNAL_FILE, as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file_csv' not in request.files or request.files['file_csv'].filename == '':
        return render_template('index.html', errore="Nessun file selezionato.", active_tab='eeg', show_detail=False)
    
    file = request.files['file_csv']
    try:
        dati = pd.read_csv(file, header=None)
        if modello:
            risultato = modello.predict(dati.T)
            previsione_finale = risultato[0]
        else:
            previsione_finale = 1 # Mock per test
            
        # Dopo la previsione mostriamo direttamente il dettaglio
        return render_template('index.html', previsione=previsione_finale, errore=None, active_tab='eeg', show_detail=True)
    except Exception as e:
        return render_template('index.html', errore=f"C'è stato un problema col file: {str(e)}", previsione=None, active_tab='eeg', show_detail=False)

# --- ROTTE MEDICO ---
@app.route('/doctor', methods=['GET'])
def doctor_home():
    # active_tab='eeg' fa aprire subito l'area di analisi
    return render_template('doctor.html', previsione=None, errore=None, active_tab='dashboard')

@app.route('/doctor_predict', methods=['POST'])
def doctor_predict():
    if 'file_csv' not in request.files or request.files['file_csv'].filename == '':
        return render_template('doctor.html', errore="Nessun file selezionato.", active_tab='eeg')
    
    file = request.files['file_csv']
    try:
        start_time = time.time() # INIZIO CRONOMETRO AI
        
        dati = pd.read_csv(file, header=None)
        if modello:
            risultato = modello.predict(dati.T)
            previsione_finale = risultato[0]
        else:
            time.sleep(0.4) # Simuliamo un piccolo ritardo se non hai il modello caricato
            previsione_finale = 1 # Mock per test
            
        end_time = time.time() # FINE CRONOMETRO AI
        tempo_analisi = round(end_time - start_time, 3) # Arrotondiamo a 3 decimali
            
        # Passiamo la variabile 'tempo_analisi' al template html
        return render_template('doctor.html', previsione=previsione_finale, errore=None, active_tab='eeg', tempo_analisi=tempo_analisi)
    except Exception as e:
        return render_template('doctor.html', errore=f"C'è stato un problema col file: {str(e)}", previsione=None, active_tab='eeg')
    
@app.route('/save_report', methods=['POST'])
def save_report():
    # Qui potrai gestire il salvataggio su database del referto del medico
    referto = request.form.get('referto')
    diagnosi = request.form.get('diagnosi')
    # Simuliamo il salvataggio e torniamo alla home del medico con un messaggio di successo
    return render_template('doctor.html', previsione=None, errore=None, active_tab='eeg', success="Referto salvato e inviato al paziente con successo!")

if __name__ == '__main__':
    porta = 5002

    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        scelta = input("Quale versione vuoi aprire? (p = paziente, d = dottore): ").strip().lower()
        percorso = '/doctor' if scelta == 'd' else '/'
        threading.Timer(1.0, lambda: webbrowser.open(f'http://127.0.0.1:{porta}{percorso}')).start()

    app.run(debug=True, port=porta)
