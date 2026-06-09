"""
Entrypoint wrapper for Streamlit Cloud / Hugging Face Spaces.
This simply calls the existing `streamlit_app.main()` so both
platforms will run the same application.
"""
from streamlit_app import main


if __name__ == "__main__":
    main()
