import streamlit as st 


def login_auth(title,func):
    if title:
        st.title(f"{title}")
    if st.session_state.authentication_status == False:
        st.error('Username/password is incorrect')
    elif st.session_state.authentication_status is None:
        st.warning('Please enter your username and password')
    else:
        func()