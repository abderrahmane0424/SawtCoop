import streamlit as st
from transformers import pipeline as hf_pipeline
from google import genai
import torch, json, re, tempfile, os, soundfile as sf
import numpy as np

# ── Config page ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Artisanat Darija → Fiche Produit",
    page_icon="🏺",
    layout="centered"
)

# ── Chargement du modèle ASR (une seule fois, mis en cache) ───────────────
@st.cache_resource(show_spinner="Chargement du modèle Darija...")
def charger_asr():
    return hf_pipeline(
        "automatic-speech-recognition",
        model="anaszil/whisper-large-v3-turbo-darija",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=st.secrets["HF_TOKEN"]
    )

# ── Client Gemini ─────────────────────────────────────────────────────────
@st.cache_resource
def charger_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ── Étape 1 : Audio → Texte Darija ───────────────────────────────────────
def transcrire_audio(audio_bytes: bytes) -> str:
    asr = charger_asr()

    # Sauvegarder temporairement pour soundfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Vérification et normalisation de l'audio
        audio_array, sample_rate = sf.read(tmp_path)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)   # mono

        # Rééchantillonnage à 16kHz si nécessaire
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        result = asr(
            {"array": audio_array.astype(np.float32), "sampling_rate": 16000},
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=False
        )
        return result["text"]
    finally:
        os.unlink(tmp_path)

# ── Étape 2 : Texte Darija → Fiche Produit JSON ───────────────────────────
def generer_fiche(texte_darija: str) -> dict:
    client = charger_gemini()

    prompt = f"""
Tu es un expert en marketing pour l'artisanat marocain authentique.
Une artisane a décrit son produit en darija : "{texte_darija}"

Génère une fiche produit professionnelle en JSON pour une plateforme e-commerce :
{{
    "titre": "Nom commercial précis et accrocheur",
    "description_courte": "Une phrase percutante pour la liste de produits (max 15 mots)",
    "description_detaillee": "3-4 phrases riches mettant en valeur l'authenticité et le savoir-faire",
    "points_cles": ["avantage 1", "avantage 2", "avantage 3"],
    "mots_cles_seo": ["mot1", "mot2", "mot3", "mot4", "mot5"],
    "categorie": "huile|bijou|tapis|poterie|cuir|vêtement|épice|autre",
    "materiaux": ["matériau détecté dans la description"],
    "origine": "ville ou région si mentionnée, sinon null",
    "labels_qualite": ["ex: fait main", "ex: coopérative féminine", "ex: bio", "ex: sans pesticides"],
    "public_cible": "touristes|diaspora|export_europe|marché_local",
    "prix_mentionne_par_artisane": "prix exact si dit dans l'audio, sinon null"
}}

Règles strictes :
- N'invente JAMAIS un prix si l'artisane ne l'a pas mentionné
- Extrais uniquement ce qui est dit dans la description
- Si un prix est mentionné (en DH, euros, ou autre), mets-le dans prix_mentionne_par_artisane
- Retourne UNIQUEMENT le JSON, sans markdown
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}

# ── Interface Streamlit ───────────────────────────────────────────────────
st.title("🏺 Artisanat Marocain")
st.markdown("**Décrivez votre produit en darija → fiche produit professionnelle en français**")
st.divider()

# --- Étape 1 : Capture audio
st.markdown("### 🎤 Étape 1 — Enregistrez votre description")

tab_rec, tab_upload = st.tabs(["Enregistrer", "Importer un fichier"])
audio_bytes = None

with tab_rec:
    st.caption("Décrivez votre produit : matière, origine, qualité, utilisation...")
    audio_value = st.audio_input("Appuyez pour enregistrer")
    if audio_value:
        audio_bytes = audio_value.getvalue()

with tab_upload:
    uploaded = st.file_uploader("Fichier audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(uploaded)

# --- Étape 2 : Transcription
if audio_bytes:
    st.divider()
    st.markdown("### 📝 Étape 2 — Transcription Darija")

    with st.spinner("Transcription en cours avec Whisper Darija..."):
        try:
            texte_darija = transcrire_audio(audio_bytes)
        except Exception as e:
            st.error(f"Erreur ASR : {e}")
            st.stop()

    texte_corrige = st.text_area(
        "Texte transcrit — corrigez si nécessaire avant de générer :",
        value=texte_darija,
        height=120,
        help="Le modèle peut faire des erreurs sur les noms propres ou termes techniques"
    )

    # --- Étape 3 : Génération
    st.divider()
    st.markdown("### ✨ Étape 3 — Génération de la fiche produit")

    if not texte_corrige.strip():
        st.warning("La transcription est vide. Réenregistrez ou corrigez le texte.")
    elif st.button("Générer la fiche produit", type="primary", use_container_width=True):

        with st.spinner("Génération avec Gemini..."):
            try:
                fiche = generer_fiche(texte_corrige)
            except Exception as e:
                st.error(f"Erreur Gemini : {e}")
                st.stop()

        if not fiche:
            st.error("Parsing JSON échoué. Réessayez.")
        else:
            st.success("Fiche générée avec succès !")

            # En-tête fiche
            col_titre, col_cat = st.columns([3, 1])
            with col_titre:
                st.markdown(f"## {fiche.get('titre', '')}")
            with col_cat:
                cat = fiche.get('categorie', '').upper()
                st.markdown(f"<br>**`{cat}`**", unsafe_allow_html=True)

            st.markdown(f"_{fiche.get('description_courte', '')}_")
            st.markdown(fiche.get('description_detaillee', ''))

            # Prix si l'artisane l'a mentionné
            prix = fiche.get('prix_mentionne_par_artisane')
            if prix:
                st.info(f"💰 Prix indiqué par l'artisane : **{prix}**")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**✅ Points clés**")
                for p in fiche.get('points_cles', []):
                    st.markdown(f"- {p}")

                st.markdown("**🏷️ Labels qualité**")
                for l in fiche.get('labels_qualite', []):
                    st.markdown(f"- {l}")

            with col2:
                st.markdown("**ℹ️ Détails produit**")
                if fiche.get('origine'):
                    st.markdown(f"📍 **Origine :** {fiche['origine']}")
                if fiche.get('materiaux'):
                    st.markdown(f"🧵 **Matériaux :** {', '.join(fiche['materiaux'])}")
                if fiche.get('public_cible'):
                    st.markdown(f"🎯 **Public cible :** {fiche['public_cible']}")

                st.markdown("**🔍 Mots-clés SEO**")
                mots = fiche.get('mots_cles_seo', [])
                st.markdown("  ".join([f"`{m}`" for m in mots]))

            # Données source + export
            st.divider()
            with st.expander("📄 Voir le JSON complet"):
                fiche_export = fiche.copy()
                fiche_export["texte_source_darija"] = texte_corrige
                st.json(fiche_export)

            st.download_button(
                label="📥 Télécharger la fiche (JSON)",
                data=json.dumps(fiche_export, indent=2, ensure_ascii=False),
                file_name=f"fiche_{fiche.get('categorie', 'produit')}.json",
                mime="application/json",
                use_container_width=True
            )
