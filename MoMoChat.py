import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate

def load_authenticator():

    with open('./database/user_info.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
)
    name, authentication_status, username = authenticator.login('Login', 'main')

    if not hasattr(st.session_state,"authentication_status"):
        st.session_state.authentication_status = authentication_status

    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')
        # st.title('Some content')
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

def main():
    st.set_page_config(page_title="Welcome to MoMoChat", page_icon="🤗")
    st.header("Welcome to MoMoChat. Let's chat! 🤗")
    load_authenticator()


    


if __name__ == "__main__":
    main()
