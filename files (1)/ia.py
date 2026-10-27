"""Chatex - assistente de TI com interface gráfica, tudo em um único arquivo.

Funciona de dois jeitos:
  - Com Gemini: se houver GEMINI_API_KEY (no .env ou no sistema) e internet.
  - Local/offline: usa a base de conhecimento interna (BASE_LOCAL) quando não há
    chave, não há internet ou o Gemini falha.

Instalação:
    pip install customtkinter
    pip install python-dotenv google-genai     (opcionais, só para usar o Gemini)

Execução:
    python ai.py
"""
import queue
import re
import threading
import time

import customtkinter as ctk

# Bibliotecas opcionais: se não estiverem instaladas, o programa roda só no modo local.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# ----------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------
MODELO = "gemini-2.5-flash"
TENTATIVAS = 3
ERROS_TEMPORARIOS = ("503", "429", "UNAVAILABLE", "overloaded")

INSTRUCAO = (
    "Você é o Chatex, um assistente de Tecnologia da Informação. "
    "Responda sempre em português, de forma clara e curta (até 6 linhas). "
    "Se a pergunta não for sobre tecnologia ou TI, diga educadamente que "
    "você só responde sobre assuntos de TI."
)

# ----------------------------------------------------
# BASE LOCAL (OFFLINE): (palavras-chave, resposta)
# ----------------------------------------------------
BASE_LOCAL = [
    (["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite"],
     "Olá! Sou o Chatex. Pergunte sobre programação, redes, banco de dados, segurança..."),
    (["python"],
     "Python é uma linguagem de alto nível, simples e muito usada em automação, "
     "ciência de dados, web e IA."),
    (["tkinter"],
     "Tkinter é a biblioteca padrão do Python para criar interfaces gráficas (janelas, "
     "botões, campos de texto)."),
    (["html"],
     "HTML é a linguagem de marcação que define a estrutura de uma página web."),
    (["css"],
     "CSS define a aparência das páginas: cores, fontes, espaçamentos e layout."),
    (["javascript", "js"],
     "JavaScript é a linguagem que dá interatividade às páginas web e também roda no "
     "servidor com Node.js."),
    (["php"],
     "PHP é uma linguagem de servidor muito usada na web. Para banco de dados, o ideal é "
     "usar PDO com prepared statements."),
    (["pdo"],
     "PDO é a camada do PHP para acessar bancos de dados de forma segura e independente "
     "do SGBD, usando prepared statements contra SQL Injection."),
    (["sql", "mysql", "postgres", "banco de dados"],
     "SQL é a linguagem usada para consultar e manipular bancos de dados relacionais "
     "(SELECT, INSERT, UPDATE, DELETE)."),
    (["git"],
     "Git é um sistema de controle de versão. Comandos básicos: git add, git commit, "
     "git push e git pull."),
    (["github"],
     "GitHub é uma plataforma que hospeda repositórios Git e facilita o trabalho em equipe."),
    (["ip"],
     "O endereço IP identifica um dispositivo em uma rede. Pode ser IPv4 (ex.: 192.168.0.1) "
     "ou IPv6."),
    (["dns"],
     "O DNS traduz nomes de sites (google.com) para endereços IP."),
    (["rede", "redes", "roteador", "wifi"],
     "Redes conectam dispositivos para trocar dados. Roteadores ligam redes; switches "
     "conectam dispositivos dentro de uma rede."),
    (["linux"],
     "Linux é um sistema operacional de código aberto, muito usado em servidores."),
    (["windows"],
     "Windows é o sistema operacional da Microsoft, o mais usado em desktops."),
    (["ram", "memória", "memoria"],
     "A RAM é a memória temporária do computador. Quanto mais RAM, mais programas "
     "rodam ao mesmo tempo."),
    (["cpu", "processador"],
     "A CPU é o processador, o 'cérebro' do computador, que executa as instruções "
     "dos programas."),
    (["ssd", "hd"],
     "SSD é mais rápido e silencioso que o HD, pois não tem partes móveis."),
    (["phishing"],
     "Phishing é um golpe que engana a pessoa com e-mails ou sites falsos para roubar "
     "dados. Desconfie de links e anexos inesperados."),
    (["senha", "segurança", "seguranca", "hacker"],
     "Use senhas longas e únicas, ative a verificação em duas etapas e mantenha o "
     "sistema atualizado."),
    (["api"],
     "API é uma interface que permite que sistemas conversem entre si, geralmente via "
     "HTTP e JSON."),
    (["nuvem", "cloud", "aws", "azure"],
     "Computação em nuvem é o uso de servidores e serviços pela internet, pagando "
     "conforme o uso."),
    (["ia", "inteligência artificial", "inteligencia artificial", "machine learning"],
     "IA é a área da computação que cria sistemas capazes de aprender e tomar decisões, "
     "como chatbots e reconhecimento de imagens."),
    (["java"],
     "Java é uma linguagem orientada a objetos, portável (roda na JVM) e muito usada em "
     "sistemas corporativos e Android."),
    (["bug", "debug", "erro"],
     "Bug é um erro no código. Para achar, leia a mensagem de erro, isole o trecho "
     "problemático e use prints ou o debugger."),
]

RESPOSTA_PADRAO = (
    "Ainda não sei responder isso. Tente perguntar sobre: Python, HTML, CSS, JavaScript, "
    "SQL, Git, redes, Linux, hardware, segurança ou IA."
)


def _responder_local(pergunta):
    texto = pergunta.lower()
    for palavras, resposta in BASE_LOCAL:
        for palavra in palavras:
            if re.search(r"\b" + re.escape(palavra) + r"\b", texto):
                return resposta
    return RESPOSTA_PADRAO


# ----------------------------------------------------
# ASSISTENTE (Gemini com fallback para a base local)
# ----------------------------------------------------
class Assistente:
    """Responde via Gemini quando possível; senão, usa a base local."""

    def __init__(self):
        self.client = None
        self.chat = None
        if genai is not None:
            try:
                self.client = genai.Client()
            except Exception:
                self.client = None  # sem chave configurada
        self.resetar()

    @property
    def usando_gemini(self):
        return self.chat is not None

    def resetar(self):
        """Reinicia a conversa (apaga o histórico enviado ao Gemini)."""
        self.chat = None
        if self.client is not None:
            try:
                self.chat = self.client.chats.create(
                    model=MODELO,
                    config=types.GenerateContentConfig(system_instruction=INSTRUCAO),
                )
            except Exception:
                self.chat = None

    def responder(self, pergunta):
        """Retorna (texto, fonte), onde fonte é 'Gemini' ou 'Local'."""
        if self.chat is None:
            return _responder_local(pergunta), "Local"

        ultimo_erro = ""
        for tentativa in range(TENTATIVAS):
            try:
                resposta = self.chat.send_message(pergunta)
                texto = (resposta.text or "").strip()
                if texto:
                    return texto, "Gemini"
                ultimo_erro = "resposta vazia (possível bloqueio de segurança)"
                break
            except Exception as e:
                ultimo_erro = str(e)
                temporario = any(cod in ultimo_erro for cod in ERROS_TEMPORARIOS)
                if temporario and tentativa < TENTATIVAS - 1:
                    time.sleep(2 * (tentativa + 1))  # espera 2s, depois 4s
                    continue
                break

        print(f"[Gemini] Falha: {ultimo_erro}")  # detalhe fica no terminal
        aviso = "(Gemini indisponível no momento, usando a base local)\n\n"
        return aviso + _responder_local(pergunta), "Local"


# ----------------------------------------------------
# INTERFACE GRÁFICA
# ----------------------------------------------------
fila = queue.Queue()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ChatexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Chatex")
        self.geometry("900x650")
        self.minsize(700, 500)
        self.logged_user = ""
        self.assistente = Assistente()
        self.ocupado = False

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=20, pady=20)

        self.create_login_screen()

    def clear_container(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    # ---------------- LOGIN ----------------
    def create_login_screen(self):
        self.clear_container()

        login_frame = ctk.CTkFrame(self.container, corner_radius=16, width=380, height=360)
        login_frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        title = ctk.CTkLabel(login_frame, text="Chatex", font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(pady=(35, 5))

        subtitle = ctk.CTkLabel(login_frame, text="Acesse sua conta para continuar", font=ctk.CTkFont(size=13), text_color="gray")
        subtitle.pack(pady=(0, 25))

        self.user_entry = ctk.CTkEntry(login_frame, placeholder_text="Usuário", width=280, height=45, corner_radius=8)
        self.user_entry.pack(pady=(0, 15))
        self.user_entry.focus()

        self.pass_entry = ctk.CTkEntry(login_frame, placeholder_text="Senha", show="*", width=280, height=45, corner_radius=8)
        self.pass_entry.pack(pady=(0, 25))
        self.pass_entry.bind("<Return>", lambda event: self.handle_login())

        btn_login = ctk.CTkButton(login_frame, text="Entrar", width=280, height=45, corner_radius=8, font=ctk.CTkFont(size=14, weight="bold"), command=self.handle_login)
        btn_login.pack()

    def handle_login(self):
        user = self.user_entry.get().strip()
        passw = self.pass_entry.get().strip()

        if user and passw:
            self.logged_user = user.capitalize()
            self.create_chat_screen()
        else:
            print("Preencha os campos!")

    # ---------------- CHAT ----------------
    def create_chat_screen(self):
        self.clear_container()
        self.assistente.resetar()

        top_bar = ctk.CTkFrame(self.container, height=65, corner_radius=12)
        top_bar.pack(fill="x", pady=(0, 15))
        top_bar.pack_propagate(False)

        brand_label = ctk.CTkLabel(top_bar, text="Chatex", font=ctk.CTkFont(size=18, weight="bold"))
        brand_label.pack(side="left", padx=20)

        user_label = ctk.CTkLabel(top_bar, text=f"| Usuário: {self.logged_user}", font=ctk.CTkFont(size=13), text_color="gray")
        user_label.pack(side="left", padx=5)

        self.theme_btn = ctk.CTkButton(top_bar, text="Alternar Tema", width=110, height=32, corner_radius=8, command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=20)

        modo = "Modo: Gemini" if self.assistente.usando_gemini else "Modo: local (offline)"
        modo_label = ctk.CTkLabel(top_bar, text=modo, font=ctk.CTkFont(size=12), text_color="gray")
        modo_label.pack(side="right", padx=5)

        self.chat_scroll = ctk.CTkScrollableFrame(self.container, corner_radius=12, fg_color="transparent")
        self.chat_scroll.pack(fill="both", expand=True, pady=(0, 15))

        bottom_bar = ctk.CTkFrame(self.container, height=75, corner_radius=12, fg_color="transparent")
        bottom_bar.pack(fill="x")
        bottom_bar.pack_propagate(False)

        self.msg_entry = ctk.CTkEntry(bottom_bar, placeholder_text="Digite sua mensagem sobre TI...", height=50, corner_radius=10, font=ctk.CTkFont(size=14))
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.msg_entry.bind("<Return>", self.enviar)

        self.send_btn = ctk.CTkButton(bottom_bar, text="Enviar", width=100, height=50, corner_radius=10, font=ctk.CTkFont(size=14, weight="bold"), command=self.enviar)
        self.send_btn.pack(side="left", padx=(0, 8))

        clear_btn = ctk.CTkButton(bottom_bar, text="Limpar", width=90, height=50, corner_radius=10, fg_color="#ef4444", hover_color="#dc2626", font=ctk.CTkFont(size=14, weight="bold"), command=self.limpar_chat)
        clear_btn.pack(side="left")

        self.adicionar_balao("Chatex (Assistente)", "Olá! Sou o Chatex, como posso te ajudar hoje?", is_bot=True)

    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")

    def adicionar_balao(self, autor, texto, is_bot=True):
        wrapper = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        wrapper.pack(fill="x", pady=8, padx=10)

        dark = ctk.get_appearance_mode() == "Dark"

        if is_bot:
            bubble_color = "#1e293b" if dark else "#e2e8f0"
            text_color = "#f8fafc" if dark else "#0f172a"
            anchor_side = "w"
        else:
            bubble_color = "#2563eb"
            text_color = "#ffffff"
            anchor_side = "e"

        content_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
        content_frame.pack(anchor=anchor_side)

        lbl_autor = ctk.CTkLabel(content_frame, text=autor, font=ctk.CTkFont(size=11, weight="bold"), text_color="gray")
        lbl_autor.pack(anchor=anchor_side, padx=4, pady=(0, 2))

        lbl_texto = ctk.CTkLabel(
            content_frame,
            text=texto,
            font=ctk.CTkFont(size=13),
            text_color=text_color,
            fg_color=bubble_color,
            corner_radius=12,
            padx=14,
            pady=10,
            wraplength=550,
            justify="left",
        )
        lbl_texto.pack(anchor=anchor_side)

        self.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    # ---------------- ENVIO / RESPOSTA ----------------
    def consultar(self, pergunta):
        """Roda em thread separada para não travar a janela."""
        try:
            texto, fonte = self.assistente.responder(pergunta)
            fila.put((f"Chatex ({fonte})", texto))
        except Exception as e:
            fila.put(("Chatex (Erro)", f"Erro inesperado: {e}"))

    def verificar_fila(self):
        try:
            autor, texto = fila.get_nowait()
        except queue.Empty:
            self.after(100, self.verificar_fila)
            return

        self.adicionar_balao(autor, texto, is_bot=True)
        self.ocupado = False
        self.send_btn.configure(state="normal", text="Enviar")
        self.msg_entry.focus()

    def enviar(self, event=None):
        if self.ocupado:
            return

        pergunta = self.msg_entry.get().strip()
        if not pergunta:
            return

        self.ocupado = True
        self.adicionar_balao(f"{self.logged_user} (Usuário)", pergunta, is_bot=False)
        self.msg_entry.delete(0, ctk.END)
        self.send_btn.configure(state="disabled", text="...")

        threading.Thread(target=self.consultar, args=(pergunta,), daemon=True).start()
        self.after(100, self.verificar_fila)

    def limpar_chat(self):
        if self.ocupado:
            return
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
        self.assistente.resetar()
        self.adicionar_balao("Chatex (Assistente)", "Chat limpo! Em que posso ajudar na área de TI?", is_bot=True)


if __name__ == "__main__":
    app = ChatexApp()
    app.mainloop()