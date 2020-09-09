import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open("scaling.pkl","rb"))





def predict_Parkinson_Disease(MDVPFoHz,MDVPShimmer, ShimmerAPQ3, HNR,spread1,D2,PPE):
        input=np.array([[MDVPFoHz,MDVPShimmer, ShimmerAPQ3, HNR,spread1,D2,PPE]]).astype(np.float64)
        input=scaler.transform(input)
        prediction=model.predict(input)
        return prediction[0]
def main():

    
    html_temp = """
    <div style="background-color:#1c1818 ;padding:10px">
    <h2 style="color:white;text-align:center;">Parkinson's Disease Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    MDVPFoHz = st.text_input("MDVP:Fo(Hz)")
    MDVPShimmer = st.text_input("MDVP:Shimmer")
    ShimmerAPQ3 = st.text_input("Shimmer:APQ3")
    HNR = st.text_input("HNR")
    spread1 = st.text_input("spread1")
    D2 = st.text_input("D2")
    PPE = st.text_input("PPE")
    
    no_html="""  
      <div style="background-color:None;padding:10px >
       <h2 style="color:white;text-align:center;"> Your dont have Parkinson's Disease</h2>
       </div>
    """
    yes_html="""  
      <div style="background-color:None;padding:10px >
       <h2 style="color:black ;text-align:center;"> You have Parkinson's Disease</h2>
       </div>
    """
   

    if st.button("Predict"):
        output=predict_Parkinson_Disease(MDVPFoHz,MDVPShimmer, ShimmerAPQ3, HNR,spread1,D2,PPE)
        st.markdown('The Predicted value is: {}'.format(output))

        if output==1:
            st.markdown(yes_html,unsafe_allow_html=True)
        else:
            st.markdown(no_html,unsafe_allow_html=True)

  

if __name__=='__main__':
    main()