import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
import time  # For time tracking
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

llm = ChatOllama(
    model = "nemotron-mini",
    temperature = 0,
    num_predict = 256,
    # other params ...
)

def response_generator(query):
    res = llm.invoke(query)
    return res.content


def main():
    st.title("🤖 Healthcare E-Commerce Chatbot ")
    st.markdown("""This chatbot assists users with **product information, price comparison, ordering and personalized recommendations** using specialized AI agents.""")
    st.subheader("",divider="rainbow")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #     # "New Chat" button to clear messages
    # if st.button("🆕 New Chat"):
    #     st.session_state.messages = []
    #     st.rerun()  # Refresh the app to clear messages
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": query})


        with st.chat_message("user"):
            st.markdown(query) 
        
        start_time = time.time()
        response = response_generator(query)
        # End time tracking
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Format the response with time taken
        response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.expander("📌 **You may also like ...**"):
    
            # Create three columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image("assets/youm.png", caption="youm vitamin")
            with col2:
                st.image("assets/herbal_tea.png", caption="Herbal Tea")
            with col3:
                st.image("assets/protein_powder.png", caption="Protein Powder")



if __name__ == "__main__":
    main()