# 🎥 Real-Time Color Tracker (OpenCV + Python)

Track one or more colours in a live webcam feed. Every object of each colour gets
its own labelled bounding box.

Good for learning:
- OpenCV basics (capture, windows, trackbars, mouse events)
- Colour spaces (BGR → HSV) and why hue wraps around for red
- Image masking, blurring and morphology
- Contours and bounding boxes

---

## 📌 Features

- 📷 Real-time webcam capture, with a clear error if the camera can't be opened
- 🎨 HSV colour ranges, including correct handling of red at **both** ends of the hue circle
- ⚪ White, grey and black are matched by brightness, not by a meaningless hue
- 🟩 One box **per object** (contours), with a minimum area to ignore noise
- 🧹 Gaussian blur and morphological open/close to clean the mask
- 🌈 Track several colours at once, each with its own label
- 🖱️ Click anywhere in the video to start tracking that colour
- 🎚️ Live HSV sliders to tune the active colour
- ⬛ Optional mask window showing what the detector sees
- ✅ Unit tests with pytest

---

## 🧠 Why HSV instead of RGB?

HSV (Hue, Saturation, Value) separates *what colour* something is (hue) from
*how vivid* (saturation) and *how bright* (value) it is. Lighting changes mostly
affect value, so a hue range is much more stable than an RGB range.

In OpenCV, hue runs 0–179 (degrees ÷ 2). Red sits at both ends (around 0 and
around 179), so a red range has to be split into two pieces and combined.

---

## 📂 Project Structure

```
realtime-color-tracker/
├── main.py              # CLI, webcam loop, windows, keyboard and mouse handling
├── colors.py            # Colour parsing and HSV range calculation (pure functions)
├── detection.py         # Blur, mask, morphology, contours, drawing
├── tests/               # pytest tests for colors, detection and the CLI
├── requirements.txt     # Runtime dependencies (pinned)
├── requirements-dev.txt # Runtime + test dependencies
└── pyproject.toml       # pytest configuration
```

---

## ⚙️ Installation

```bash
git clone https://github.com/thanycodes/realtime-color-tracker.git
cd realtime-color-tracker

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

python -m pip install -r requirements.txt
```

On macOS, the first run asks for camera access for your terminal app. If it was
denied before, enable it in **System Settings → Privacy & Security → Camera**.

---

## 📦 Usage

```bash
python main.py                         # track green
python main.py -c red -c blue          # track several colours
python main.py -c "#ff8000"            # hex colour (RGB order, like the web)
python main.py -c 0,255,255            # B,G,R triple (OpenCV order)
python main.py --tune --show-mask      # open the sliders and mask window
python main.py --list-colors           # show the named colours
```

### Options

| Option | Default | What it does |
|---|---|---|
| `-c`, `--color` | `green` | Colour to track: name, `#RRGGBB` or `B,G,R`. Repeat for several. |
| `--camera` | `0` | Camera index to open |
| `--tolerance` | `10` | Hue tolerance either side of the colour (0–90) |
| `--min-area` | `500` | Ignore blobs smaller than this many pixels |
| `--max-objects` | `0` | Max boxes per colour (0 = no limit) |
| `--blur` | `5` | Gaussian blur kernel size (0 = off) |
| `--kernel` | `5` | Morphology kernel size for mask clean-up (0 = off) |
| `--tune` | off | Open the HSV sliders at start-up |
| `--show-mask` | off | Open the mask window at start-up |
| `--mirror` | off | Flip the video horizontally |
| `--list-colors` | | Print named colours and exit |

Named colours: red, orange, yellow, green, cyan, blue, purple, magenta, pink,
white, black.

### Controls (while running)

| Key / action | What it does |
|---|---|
| Left-click | Pick the colour under the cursor and start tracking it |
| `n` / `Tab` | Make the next tracked colour active |
| `t` | Show/hide the tuning sliders (they edit the active colour) |
| `m` | Show/hide the mask window (shows the active colour's mask) |
| `r` | Reset the active colour's range to its default |
| `d` | Stop tracking the active colour |
| `q` / `Esc` | Quit |

### Tuning tips

- If `H low` is greater than `H high`, the hue range wraps around (this is how red works).
- Raise `S low` to ignore washed-out, greyish pixels.
- Raise `V low` to ignore shadows. Lower it in dim rooms.
- Raise `Min area` if small specks keep getting boxed.

---

## ✅ Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests use synthetic images, so they don't need a webcam.
