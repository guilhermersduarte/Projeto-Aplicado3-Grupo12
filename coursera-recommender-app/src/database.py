import streamlit as st
from supabase import create_client, Client


def get_supabase() -> Client:
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            st.secrets["supabase"]["url"],
            st.secrets["supabase"]["key"],
        )
    return st.session_state.supabase_client


def salvar_interacao(user_id: str, curso_id: str, curso_nome: str, posicao: int, modelo: str) -> None:
    supabase = get_supabase()
    supabase.table("interacoes").insert({
        "user_id": user_id,
        "curso_id": curso_id,
        "curso_nome": curso_nome,
        "posicao": posicao,
        "modelo": modelo,
    }).execute()


def salvar_feedback(user_id: str, curso_id: str, nota: int) -> None:
    nota = max(1, min(5, int(nota)))
    supabase = get_supabase()
    supabase.table("feedback").insert({
        "user_id": user_id,
        "curso_id": curso_id,
        "nota": nota,
    }).execute()


def salvar_feedback_qualitativo(user_id: str, opcao: str, comentario: str, total_avaliacoes: int) -> None:
    supabase = get_supabase()
    supabase.table("feedback_qualitativo").insert({
        "user_id": user_id,
        "opcao": opcao,
        "comentario": comentario or None,
        "total_avaliacoes": total_avaliacoes,
    }).execute()


def buscar_historico(user_id: str) -> list:
    supabase = get_supabase()
    result = supabase.table("feedback").select("curso_id, nota").eq("user_id", user_id).limit(200).execute()
    return result.data
