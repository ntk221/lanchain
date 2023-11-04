import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

load_dotenv()

def create_agent_chain():
    chat = ChatOpenAI(
            model_name=os.environ["OPENAI_API_MODEL"],
            temperature=os.environ["OPENAI_API_TEMPERATURE"],
            streaming=True,
        )
    
    # OpenAI Functions Agent のプロンプトに Memory の会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    # OpenAI Functions Agent が使える設定でMemoruを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        
    tools = load_tools({"ddg-search", "wikipedia"})
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("langchain-streamlit-app")

if "messages" not in st.session_state: # st.session_state に messageがないとき
    st.session_state.messages = [] # st.session_state.messagesをからのリストで初期化
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # ロールごとに
        st.markdown(message["content"]) # 保存されているテキストを表示
        
    
prompt = st.chat_input("What is up?")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    
