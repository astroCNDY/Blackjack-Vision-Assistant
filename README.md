

# Blackjack Vision Assistant

This project is a simple Python-based blackjack helper that supports two modes:

1. **Normal Mode** – Displays a UI to add, edit, and manage cards for both the player and the dealer.
2. **Camera Mode** – Replaces the UI with a live webcam feed (including iPhone Continuity Camera and Desk View on macOS) and allows capturing a frame of the table.

The repository contains the following files:
- `blackjack_core.py` – Contains blackjack logic and card-handling functions.
- `gui.py` – Contains the GUI, camera mode switching, camera preview, and capture functionality.
- `requirements.txt` – List of dependencies required to run the project.
- `.env` – Used to store `OPENAI_API_KEY=` (kept empty by default).

---

## Requirements

This project requires Python 3.10 or newer.

Install all dependencies using:

```bash
pip install -r requirements.txt
````

---

## Environment Variables

A `.env` file should exist in the project folder with the following content:

```
OPENAI_API_KEY=
```

Your OpenAI API key should be added to this file to be able to use the AI functionality.  

---

## Running the Application

Run the program with:

```bash
python gui.py
```

Inside the application:

* Open **Settings** to switch between **Normal Mode** and **Camera Mode**.
* Select the available camera device from the list (supports webcams, USB cameras, Continuity Camera, and Desk View on macOS (Based on your Mac OS version)).

---

## Notes

* Only the `.env` file should contain the API key.
* `blackjack_core.py` and `gui.py` must stay in the same directory.
* The `requirements.txt` file ensures all dependencies can be installed easily.
* The application assumes the operating system already has camera permissions configured.

---

