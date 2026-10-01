"""Entry point. Run with:

    python app.py

Opens the Gradio UI automatically in your default browser.
"""

from ui.blocks import demo

if __name__ == "__main__":
    demo.launch(inbrowser=True)
