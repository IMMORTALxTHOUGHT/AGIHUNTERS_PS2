"""ForgeMind dashboard entry point.

`python3 -m dashboard` launches the standalone Forge HUD (custom inspection
terminal) by default. The Gradio app remains available via
`python3 -m dashboard.app`.
"""
import dashboard.serve as serve

if __name__ == "__main__":
    serve.launch()
