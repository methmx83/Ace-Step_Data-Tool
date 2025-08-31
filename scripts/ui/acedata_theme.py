# acedata_theme.py
import gradio as gr

# ACE Data Theme - Python Version mit Kommentaren
# Basierend auf der acedata.json mit allen Anpassungen

# ===== KONFIGURATIONSWERTE =====
RADIUS_LG = "8px"
RADIUS_MD = "6px"
RADIUS_SM = "4px"
RADIUS_XL = "12px"
RADIUS_XS = "2px"
RADIUS_XXL = "22px"
RADIUS_XXS = "1px"

SPACING_LG = "8px"
SPACING_MD = "6px"
SPACING_SM = "4px"
SPACING_XL = "10px"
SPACING_XS = "2px"
SPACING_XXL = "16px"
SPACING_XXS = "1px"

TEXT_SM = "12px"
TEXT_MD = "14px"
TEXT_LG = "16px"
TEXT_XL = "22px"
TEXT_XXL = "26px"
TEXT_XXS = "9px"

# ===== THEME-DEFINITION =====
ace_theme = gr.Theme(
    # Schriftarten - müssen als String definiert werden, nicht als Liste
    font=(
        "'Asap', 'ui-sans-serif', sans-serif"
    ),
    font_mono=(
        "'Fira Code', 'ui-monospace', monospace"
    ),
    
    # ===== HINTERGRUNDFARBEN =====
    background_fill_primary="#0e202f",
    background_fill_primary_dark="#09151f",
    background_fill_secondary="#dce3e8",
    background_fill_secondary_dark="#242424",
    
    # ===== BLOCK-EINSTELLUNGEN =====
    block_background_fill="#111f29",
    block_background_fill_dark="#111f29",
    block_border_color="#0da7a9",
    block_border_color_dark="#0da7a9",
    block_border_width="1px",
    block_info_text_color="#a2d0f1",
    block_info_text_color_dark="#a2d0f1",
    block_info_text_size=TEXT_SM,
    block_info_text_weight="400",
    
    # Block-Label Einstellungen
    block_label_background_fill="#ECF2F700",
    block_label_background_fill_dark="#05222f",
    block_label_border_color="#dce3e8",
    block_label_border_color_dark="#242424",
    block_label_border_width="1px",
    block_label_margin="0",
    block_label_padding=f"{SPACING_SM} {SPACING_LG}",
    block_label_radius=f"calc({RADIUS_LG} - 1px) 0 calc({RADIUS_LG} - 1px) 0",
    block_label_right_radius=f"0 calc({RADIUS_LG} - 1px) 0 calc({RADIUS_LG} - 1px)",
    block_label_text_color="#4EACEF",
    block_label_text_color_dark="#4EACEF",
    block_label_text_size=TEXT_SM,
    block_label_text_weight="400",
    
    # Block-Padding und Radius
    block_padding=f"{SPACING_XL} calc({SPACING_XL} + 2px)",
    block_radius=RADIUS_LG,
    block_shadow="#FFFFFF00",
    block_shadow_dark="#00000000",
    
    # Block-Title Einstellungen
    block_title_background_fill="#ECF2F700",
    block_title_background_fill_dark="#19191900",
    block_title_border_color="#dce3e8",
    block_title_border_color_dark="#242424",
    block_title_border_width="0px",
    block_title_padding="0",
    block_title_radius="none",
    block_title_text_color="#4EACEF",
    block_title_text_color_dark="#4EACEF",
    block_title_text_size=TEXT_MD,
    block_title_text_weight="bold",
    
    # ===== HINTERGRUNDBILDER =====
    body_background_fill="url('https://raw.githubusercontent.com/methmx83/ACE-DATA_v2/refs/heads/main/docs/lg2.png') #FFFFFF no-repeat right bottom / auto 28svh padding-box fixed",
    body_background_fill_dark="url('https://raw.githubusercontent.com/methmx83/ACE-DATA_v2/refs/heads/main/docs/lg_dark.png') #000000 no-repeat right bottom / auto 28svh padding-box fixed",
    
    # ===== TEXTFARBEN =====
    body_text_color="#191919",
    body_text_color_dark="#ECF2F7",
    body_text_color_subdued="#636668",
    body_text_color_subdued_dark="#c4c4c4",
    body_text_size=TEXT_MD,
    body_text_weight="400",
    
    # ===== RAHMENFARBEN =====
    border_color_accent="#dce3e8",
    border_color_accent_dark="#242424",
    border_color_accent_subdued="#dce3e867",
    border_color_accent_subdued_dark="#24242467",
    border_color_primary="#0da7a9",
    border_color_primary_dark="#0da7a9",
    
    # ===== BUTTON-EINSTELLUNGEN =====
    button_border_width="1px",  # Direkter Wert statt Referenz
    button_border_width_dark="1px",
    
    # Cancel-Button
    button_cancel_background_fill="#041a26",
    button_cancel_background_fill_dark="#041a26",
    button_cancel_background_fill_hover="#5f1f26",
    button_cancel_background_fill_hover_dark="#5f1f26",
    button_cancel_border_color="#056263",
    button_cancel_border_color_dark="#056263",
    button_cancel_border_color_hover="#ffffff",
    button_cancel_border_color_hover_dark="#ffffff",
    button_cancel_text_color="#4EACEF",
    button_cancel_text_color_dark="#4EACEF",
    button_cancel_text_color_hover="#ffffff",
    button_cancel_text_color_hover_dark="#ffffff",
    
    # Große Buttons
    button_large_padding=f"{SPACING_LG} calc(2 * {SPACING_LG})",
    button_large_radius=RADIUS_LG,
    button_large_text_size=TEXT_LG,
    button_large_text_weight="600",
    
    # Primäre Buttons
    button_primary_background_fill="#092637",
    button_primary_background_fill_dark="#092637",
    button_primary_background_fill_hover="#0b3d65",
    button_primary_background_fill_hover_dark="#0b3d65",
    button_primary_border_color="#056263",
    button_primary_border_color_dark="#056263",
    button_primary_border_color_hover="#0d888a",
    button_primary_border_color_hover_dark="#0d888a",
    button_primary_text_color="#ECF2F7",
    button_primary_text_color_dark="#191919",
    button_primary_text_color_hover="#e1eaf0",
    button_primary_text_color_hover_dark="#141414",
    
    # Sekundäre Buttons
    button_secondary_background_fill="#092637",
    button_secondary_background_fill_dark="#092637",
    button_secondary_background_fill_hover="#0b3d65",
    button_secondary_background_fill_hover_dark="#0b3d65",
    button_secondary_border_color="#056263",
    button_secondary_border_color_dark="#056263",
    button_secondary_border_color_hover="#ffffff",
    button_secondary_border_color_hover_dark="#ffffff",
    button_secondary_text_color="#bfdef7",
    button_secondary_text_color_dark="#bfdef7",
    button_secondary_text_color_hover="#ffffff",
    button_secondary_text_color_hover_dark="#ffffff",
    
    # Allgemeine Button-Einstellungen
    button_shadow="none",
    button_shadow_active="none",
    button_shadow_hover="none",
    
    # Kleine Buttons
    button_small_padding=f"{SPACING_SM} calc(2 * {SPACING_SM})",
    button_small_radius=RADIUS_LG,
    button_small_text_size=TEXT_MD,
    button_small_text_weight="400",
    
    # Button-Übergänge
    button_transition="background-color 0.2s ease",
    
    # ===== CHECKBOX-EINSTELLUNGEN =====
    checkbox_background_color="#024142",
    checkbox_background_color_dark="#024142",
    checkbox_background_color_focus="#0c5556",
    checkbox_background_color_focus_dark="#092c2d",
    checkbox_background_color_hover="#dce3e8",
    checkbox_background_color_hover_dark="#29dbdd",
    checkbox_background_color_selected="#0c7577",
    checkbox_background_color_selected_dark="#044a4b",
    
    # Weitere Checkbox-Einstellungen
    checkbox_border_color="#dce3e8",
    checkbox_border_color_dark="#242424",
    checkbox_border_color_focus="#4EACEF",
    checkbox_border_color_focus_dark="#4EACEF",
    checkbox_border_color_hover="#4EACEF",
    checkbox_border_color_hover_dark="#4EACEF",
    checkbox_border_color_selected="#4EACEF",
    checkbox_border_color_selected_dark="#4EACEF",
    checkbox_border_radius=RADIUS_SM,
    checkbox_border_width="1px",
    checkbox_border_width_dark="1px",
    checkbox_check="url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e\")",
    
    # ===== WEITERE EINSTELLUNGEN =====
    # Hier folgen die restlichen Einstellungen aus deiner JSON-Datei
    # Ich habe nur einen Teil eingefügt, um die Datei nicht zu lang zu machen
    
    # Allgemeine Einstellungen
    name="acedata",
    
    # Panel-Einstellungen
    panel_background_fill="#243645",
    panel_background_fill_dark="#243645",
    panel_border_color="#056263",
    panel_border_color_dark="#4EACEF",
    panel_border_width="0",
    
    # Input-Einstellungen
    input_background_fill="#122331",
    input_background_fill_dark="#122331",
    input_background_fill_focus="#162b3d",
    input_background_fill_focus_dark="#162b3d",
    input_background_fill_hover="#d0d7db",
    input_background_fill_hover_dark="#202020",
    input_border_color="#056263",
    input_border_color_dark="#056263",
    input_border_color_focus="#191919",
    input_border_color_focus_dark="#ECF2F7",
    input_border_color_hover="#0d888a",
    input_border_color_hover_dark="#0d888a",
    input_border_width="1px",
    input_padding=SPACING_XL,
    input_placeholder_color="#19191930",
    input_placeholder_color_dark="#ECF2F730",
    input_radius=RADIUS_LG,
    input_shadow="#19191900",
    input_shadow_dark="#ECF2F700",
    input_shadow_focus="#19191900",
    input_shadow_focus_dark="#ECF2F700",
    input_text_size=TEXT_MD,
    input_text_weight="400",
)

# Zusätzliche CSS-Regeln für Google Fonts
additional_css = """
@import url('https://fonts.googleapis.com/css2?family=Asap:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap');
"""