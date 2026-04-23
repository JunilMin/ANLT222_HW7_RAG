# Yosemite RAG Project

## Setup
1. Put your OpenAI key in a `.env` file:
   OPENAI_API_KEY=your_actual_key_here

2. Install dependencies:
   pip install -r requirements.txt

3. Build the FAISS index locally:
   python preprocess_pdf.py

4. Run the terminal chat:
   python cli_chat.py

5. Run the Streamlit app:
   streamlit run app.py
   or if you are in Windows and the code is not working, try
   python -m streamlit run app.py
   