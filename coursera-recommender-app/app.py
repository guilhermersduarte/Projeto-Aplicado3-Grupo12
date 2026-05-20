import streamlit as st
from src.database import get_supabase, salvar_interacao, salvar_feedback, salvar_feedback_qualitativo, buscar_historico
from src.hybrid_recommender import HybridRecommender


@st.cache_resource
def carregar_recomendador() -> HybridRecommender:
    rec = HybridRecommender(svd_fold_in_threshold=10)
    rec.load(
        svd_path="data/models/svd_model.pkl",
        tfidf_matrix_path="data/tfidf_matrix.npz",
        tfidf_meta_path="data/tfidf_meta.parquet",
        tfidf_params_path="data/tfidf_params.json",
    )
    return rec


st.set_page_config(page_title="Recomendador de Cursos", page_icon="🎓")
st.title("Recomendador de Cursos Coursera")


# --- Auth ---

def enviar_otp(email: str) -> bool:
    try:
        get_supabase().auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
        return True
    except Exception as e:
        st.error(f"Erro ao enviar código: {e}")
        return False


def verificar_otp(email: str, token: str) -> bool:
    try:
        get_supabase().auth.verify_otp({"email": email, "token": token, "type": "email"})
        return True
    except Exception as e:
        st.error(f"Código inválido ou expirado: {e}")
        return False


def verificar_sessao():
    try:
        return get_supabase().auth.get_session()
    except Exception:
        return None


# --- Login screen ---

session = verificar_sessao()

if not session:
    if "aguardando_codigo" not in st.session_state:
        st.session_state.aguardando_codigo = False
    if "email_enviado" not in st.session_state:
        st.session_state.email_enviado = ""

    if not st.session_state.aguardando_codigo:
        st.write(
            "Este app sugere cursos do Coursera com base nos seus interesses. "
            "Faz parte de um projeto para a Universidade Mackenzie. "
            "Sua participação ajuda a testar o sistema na prática."
        )
        st.write(
            "Leva uns 5 minutos. Seu email serve só para enviar o código de acesso "
            "e evitar avaliações repetidas — não compartilhamos com ninguém."
        )
        st.subheader("Entrar")
        st.write("Digite seu email para receber um código de acesso.")
        with st.form("form_email"):
            email = st.text_input("Email", placeholder="seu@email.com")
            enviado = st.form_submit_button("Enviar código")
        if enviado:
            if not email or "@" not in email:
                st.warning("Por favor, informe um email válido.")
            elif enviar_otp(email):
                st.session_state.aguardando_codigo = True
                st.session_state.email_enviado = email
                st.rerun()
    else:
        email = st.session_state.email_enviado
        st.info(f"Código enviado para **{email}**. Verifique sua caixa de entrada.")
        with st.form("form_codigo"):
            codigo = st.text_input("Código recebido por email", max_chars=8, placeholder="57643358")
            verificar = st.form_submit_button("Verificar")
        if verificar:
            if not codigo.isdigit() or not (6 <= len(codigo) <= 8):
                st.warning("Digite o código numérico recebido no email.")
            elif verificar_otp(email, codigo):
                st.session_state.aguardando_codigo = False
                st.rerun()
        if st.button("Usar outro email"):
            st.session_state.aguardando_codigo = False
            st.rerun()

    st.stop()


# --- App autenticado ---

user_id = session.user.id
user_email = session.user.email

col1, col2 = st.columns([4, 1])
with col1:
    st.caption(f"Autenticado como **{user_email}**")
with col2:
    if st.button("Sair"):
        get_supabase().auth.sign_out()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.divider()

rec = carregar_recomendador()

historico = buscar_historico(user_id)
ratings = {h["curso_id"]: float(h["nota"]) for h in historico}
status = rec.user_status(ratings)
cursos_avaliados = set(ratings.keys())

if "numero_rodadas" not in st.session_state:
    st.session_state.numero_rodadas = 0

OPCAO_LABELS = {
    "Topo uma chamada de vídeo rápida (15 min) para conversar sobre o app": "videocall",
    "Prefiro deixar minha opinião por escrito aqui mesmo": "escrito",
    "Por enquanto não, obrigado": "nao_obrigado",
}

STAGE_LABELS = {
    "cold_start": "Baseado nas suas preferências declaradas",
    "tfidf_profile": "Baseado nos cursos que você avaliou",
    "svd_fold_in": "Recomendação personalizada (modelo colaborativo)",
}


def nome_curso(cid: str) -> str:
    return cid.replace("-", " ").title()


def gerar_recomendacoes() -> tuple[list[tuple[str, float]], str]:
    stage = status["stage"]

    if stage == "cold_start":
        return [], "cold_start"

    recs_raw = rec.recommend(ratings, top_n=50, exclude=cursos_avaliados)
    recs_raw = recs_raw[:5]
    modelo = "svd" if stage == "svd_fold_in" else "tfidf"
    return recs_raw, modelo


# --- Geração de recomendações para usuários com histórico ---

if status["stage"] != "cold_start" and "recomendacoes" not in st.session_state:
    recs_raw, modelo = gerar_recomendacoes()
    recs = [{"course_id": cid, "course_name": nome_curso(cid), "score": score} for cid, score in recs_raw]
    for i, r in enumerate(recs):
        salvar_interacao(user_id, r["course_id"], r["course_name"], i + 1, modelo)
    st.session_state.recomendacoes = recs
    st.session_state.modelo_usado = modelo
    st.session_state.stage_usado = status["stage"]
    st.rerun()

# --- Cold start: formulário de preferências ---

if status["stage"] == "cold_start" and "recomendacoes" not in st.session_state:
    faltam = status["ratings_to_fold_in"]
    st.subheader("O que você quer aprender?")
    st.write(
        "Escreva palavras-chave dos temas que te interessam. "
        "Use termos em **inglês** sempre que possível — quase todo o catálogo do Coursera "
        "está em inglês, e a busca funciona melhor assim."
    )
    st.write(
        "O sistema procura primeiro pelo nome dos cursos. Conforme você avalia, "
        "ele passa a usar também o histórico de avaliações de outros usuários para refinar as sugestões."
    )
    if faltam > 0:
        st.info(f"Avalie mais **{faltam}** curso(s) para ativar o modelo colaborativo.")
    with st.form("form_preferencias"):
        preferencias = st.text_area(
            "Seus interesses",
            placeholder="Ex: machine learning, python, finance, project management",
            height=120,
            max_chars=500,
        )
        buscar = st.form_submit_button("Ver recomendações")
    if buscar:
        if not preferencias.strip():
            st.warning("Descreva pelo menos um interesse para continuar.")
        else:
            recs_raw = rec.search_by_text(preferencias, top_n=5, exclude=cursos_avaliados)
            recs = [{"course_id": cid, "course_name": nome_curso(cid), "score": score} for cid, score in recs_raw]
            for i, r in enumerate(recs):
                salvar_interacao(user_id, r["course_id"], r["course_name"], i + 1, "tfidf")
            st.session_state.recomendacoes = recs
            st.session_state.modelo_usado = "tfidf"
            st.session_state.stage_usado = "cold_start"
            st.rerun()
    st.stop()


# --- Exibição dos cursos e coleta de feedback ---

if not st.session_state.get("feedback_salvo"):
    stage_usado = st.session_state.get("stage_usado", "cold_start")
    st.subheader("Cursos recomendados para você")
    st.caption(STAGE_LABELS.get(stage_usado, ""))
    st.write(
        "Arraste a barra de cada curso de acordo com o seu interesse real em fazê-lo. "
        "**1** = não tenho interesse, **5** = faria com certeza. "
        "Seja honesto — é assim que o sistema aprende."
    )

    with st.form("form_feedback"):
        notas = {}
        for i, rec_item in enumerate(st.session_state.recomendacoes):
            st.markdown(f"**{i + 1}. {rec_item['course_name']}**")
            notas[rec_item["course_id"]] = st.slider(
                "Seu interesse neste curso",
                min_value=1,
                max_value=5,
                value=3,
                key=f"nota_{rec_item['course_id']}",
                help="1 = nenhum interesse · 5 = muito interesse",
            )
            if i < len(st.session_state.recomendacoes) - 1:
                st.divider()
        enviar = st.form_submit_button("Enviar avaliações")

    if enviar:
        for curso_id, nota in notas.items():
            salvar_feedback(user_id, curso_id, nota)
        st.session_state.feedback_salvo = True
        st.session_state.numero_rodadas += 1
        st.rerun()

else:
    st.success("Avaliações registradas. Obrigado!")

    n_avaliados = len(ratings)
    n_rodadas = st.session_state.numero_rodadas
    META_RODADAS = 3
    META_CURSOS = 15
    limiar_atingido = n_rodadas >= META_RODADAS or n_avaliados >= META_CURSOS

    novo_status = rec.user_status({**ratings, **{r["course_id"]: 3 for r in st.session_state.get("recomendacoes", [])}})
    if novo_status["stage"] == "svd_fold_in" and status["stage"] != "svd_fold_in":
        st.info("Você ativou o modelo colaborativo! As próximas recomendações usarão seu perfil completo.")

    if not limiar_atingido:
        st.write(
            f"Progresso: **{n_rodadas} de {META_RODADAS} rodadas** · "
            f"**{n_avaliados} de {META_CURSOS} cursos avaliados**."
        )
        st.write(
            "Que tal mais uma rodada? A cada ciclo as recomendações ficam mais próximas "
            "do seu perfil, e é exatamente isso que estamos testando."
        )
        if st.button("Ver novas recomendações"):
            for key in ["recomendacoes", "modelo_usado", "stage_usado", "feedback_salvo"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        if not st.session_state.get("qualitativo_enviado"):
            st.divider()
            st.subheader("Pode nos ajudar mais um pouco?")
            st.write(
                f"Você completou **{n_rodadas} rodadas** e avaliou **{n_avaliados} cursos** "
                "— obrigado! Antes de encerrar, sua opinião sobre a experiência vale muito "
                "para o projeto."
            )

            with st.form("form_qualitativo"):
                opcao_label = st.radio(
                    "Como prefere contribuir?",
                    options=list(OPCAO_LABELS.keys()),
                    index=None,
                )
                comentario = st.text_area("Deixe seu comentário (opcional)", height=100)
                enviar_qualitativo = st.form_submit_button("Enviar")

            if enviar_qualitativo and opcao_label:
                salvar_feedback_qualitativo(
                    user_id=user_id,
                    opcao=OPCAO_LABELS[opcao_label],
                    comentario=comentario,
                    total_avaliacoes=n_avaliados,
                )
                st.session_state.ultima_opcao_qualitativa = OPCAO_LABELS[opcao_label]
                st.session_state.qualitativo_enviado = True
                st.rerun()
        else:
            st.divider()
            st.subheader("Obrigado por participar!")
            st.write(
                "Sua contribuição foi registrada e ajuda muito o projeto. "
                "Pode encerrar a sessão por aqui."
            )
            opcao_escolhida = st.session_state.get("ultima_opcao_qualitativa")
            if opcao_escolhida in {"videocall", "escrito"}:
                st.write("Entraremos em contato em breve.")
