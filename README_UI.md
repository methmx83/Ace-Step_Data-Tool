# WebUI (Gradio) — Demo

Eine kleine Gradio-Demo wurde hinzugefügt unter `scripts/ui/app.py`. Sie ist minimalistisch und dient dazu, Presets auszuwählen und Tag-Processing lokal zu testen (keine LLM-Calls).

Voraussetzungen:
- Gradio installieren: `pip install gradio` (siehe `requirements.txt`)

Starten:

```bat
python scripts\ui\app.py
```

Die UI bietet:
- Dropdown zur Auswahl eines Presets (aus `presets/`)
- Checkbox zum optionalen Aktivieren von `rap_style`
- Textfeld für Roh-Tags (Komma-separiert) zum Testen der Pipeline

Hinweis: Die Demo ist als Entwicklungs-Tool gedacht. Für produktive WebUIs bitte `scripts/ui/app.py` als Referenz nutzen.
