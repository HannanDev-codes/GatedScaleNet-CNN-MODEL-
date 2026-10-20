# 🐱🐶 Cat vs Dog Detector (CNN + Gradio)

A small project that teaches a computer to look at a picture and say **"cat"** or **"dog"**, using a
**CNN (Convolutional Neural Network)**, with a simple web page (Gradio) where you upload a photo and see the answer.

It also draws a **heatmap** showing *where* the network looked, so you can see whether it is really
looking at the face.

---

## 1. The idea in simple words

**What is a CNN?**
Imagine you slide a tiny magnifying glass across a photo. At first it only notices simple things: edges and colours.
The next layer looks at *those* findings and spots bigger things: fur texture, an eye, an ear.
The next layer combines those into a nose, a muzzle, a whole face. At the end, the network says:
"these face-like shapes look 78% like a dog".

```
Photo ─► [find edges] ─► [find textures] ─► [find eyes/ears/nose] ─► [find whole face] ─► cat or dog?
          Conv block 1     Conv block 2        Conv block 3            Conv block 4        decision
```

**How does it learn?**
1. Show it a picture. It guesses (at first, randomly).
2. Compare the guess with the true label → a number called the **loss** (how wrong it was).
3. Nudge all its internal numbers slightly so next time it's less wrong.
4. Repeat for every picture, many times. One full pass over all pictures is an **epoch**.

### Words you will see

| Word | Meaning |
|---|---|
| **Training set** | Pictures the model learns from (`data/train`) |
| **Validation set** | Pictures it has *never learned from*, used as a fair exam (`data/val`) |
| **Epoch** | One full pass over the training pictures |
| **Batch** | A small group of pictures processed together (32 by default) |
| **Loss** | How wrong the model is. Lower is better |
| **Accuracy** | % of pictures answered correctly |
| **Overfitting** | Model memorises the training pictures but fails on new ones (train accuracy high, validation low) |
| **Data augmentation** | Randomly flipping/rotating/re-colouring training pictures so the model sees more variety |
| **Dropout** | Randomly switching off parts of the network while training so it can't just memorise |
| **Grad-CAM** | The trick that makes the heatmap |

---

## 2. About the dataset (important!)

Your zip contains **345 photos**:

| | cat | dog | total |
|---|---|---|---|
| train | 95 | 180 | 275 |
| val | 24 | 46 | 70 |

Two things to know:

* **There are no face boxes.** Each image only has a label (cat or dog), so this project *classifies* the image
  ("is this a cat or a dog?"). It does not draw a rectangle around a face. The heatmap is the closest thing: it
  highlights the area the model used. (To draw real boxes you would need a labelled-box dataset and an object
  detector such as YOLO; see "Next steps".)
* **It is small and unbalanced** (about 2 dogs per cat). The code handles imbalance with *class weights*
  and handles the small size with *data augmentation*.

---

## 3. Project layout

```
cats-vs-dogs-cnn/
├── app.py                  # Gradio web app (the frontend)
├── train.py                # trains the CNN and saves the best model
├── evaluate.py             # scores the model, makes a confusion matrix
├── predict.py              # predict one image from the terminal
├── catsdogs/
│   ├── model.py            # the CNN itself (SimpleCNN) + optional ResNet18
│   ├── data.py             # loads images, augmentation, class weights
│   └── inference.py        # loads a trained model, predicts, builds the heatmap
├── data/                   # put the dataset here (see data/README.md)
├── models/                 # trained model is saved here (best_model.pt)
├── outputs/                # plots are saved here
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 4. Setup

You need Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Unzip the dataset into `data/` so you have `data/train/cat`, `data/train/dog`, `data/val/cat`, `data/val/dog`
(details in [data/README.md](data/README.md)).

No GPU? That's fine. 275 small images train in a few minutes on a normal laptop CPU.

---

## 5. Train

```bash
python train.py
```

You'll see one line per epoch:

```
Epoch  12/40 | train loss 0.512 acc 76.4% | val loss 0.498 acc 78.6%  <- saved best
```

The best model (by validation accuracy) is saved to `models/best_model.pt` and training curves to
`outputs/training_curves.png`.

**How to read the curves**
* Both losses going down → it's learning.
* Train accuracy far above validation accuracy → overfitting. Try more augmentation, more epochs with a lower
  learning rate, or more data.

### Want higher accuracy? Use a pretrained network

A CNN trained from scratch on only 275 images has limited accuracy. A pretrained ResNet18 already knows how to
"see" (it learned from millions of images), so it only has to learn "cat vs dog":

```bash
python train.py --arch resnet18 --pretrained --img-size 224 --epochs 15 --lr 3e-4
```

(Needs internet once, to download the pretrained weights.)

### Useful options

| Option | Default | What it does |
|---|---|---|
| `--epochs` | 40 | How many passes over the data |
| `--img-size` | 128 | Pictures are resized to this square size |
| `--batch-size` | 32 | Pictures per step (lower it if you run out of memory) |
| `--lr` | 0.001 | Learning rate: how big each nudge is |
| `--patience` | 12 | Stop early if no improvement for this many epochs |
| `--arch` | simple | `simple` (our CNN) or `resnet18` |
| `--pretrained` | off | Start ResNet18 from pretrained weights |

---

## 6. Evaluate

```bash
python evaluate.py
```

Prints accuracy plus precision/recall for each class, and saves `outputs/confusion_matrix.png`
(a 2×2 table: how many cats were called cats, cats called dogs, and so on).

---

## 7. Run the Gradio app

```bash
python app.py
```

Open **http://127.0.0.1:7860**, drop in a photo, and you'll get:

* the probability for **cat** and **dog**
* a **heatmap** of where the model looked

Add `--share` for a temporary public link you can send to a friend.

Prefer the terminal? `python predict.py my_photo.jpg --heatmap outputs/heat.png`

---

## 8. Push to GitHub

```bash
git init
git add .
git commit -m "Cat vs Dog CNN with Gradio app"
git branch -M main
git remote add origin https://github.com/<your-username>/cats-vs-dogs-cnn.git
git push -u origin main
```

The trained model, plots and dataset are in `.gitignore`, so the repo stays small. If you want to share the
model, use [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) or
[Hugging Face Spaces](https://huggingface.co/spaces) (Gradio apps run there for free).

---

## 9. Limitations and honest notes

* Small dataset → results vary from run to run; validation has only 70 images, so one image is ≈1.4 %.
* The model only knows cats and dogs. Show it a horse and it will still answer "cat" or "dog".
* It classifies the whole photo; it does not locate faces with boxes.
* Best results come from clear, well-lit photos where the animal's face is visible.

## 10. Next steps

* Train with `--arch resnet18 --pretrained` for much higher accuracy.
* Add more images (even 500 per class helps a lot).
* Real **face detection** with boxes: label faces with a tool like Roboflow/LabelImg and train a YOLO model.
* Predict the **breed**: the file names in the dataset contain breeds (e.g. `Persian_139_...`), so you could build a
  multi-class version.

## License

MIT, see [LICENSE](LICENSE).

## A third architecture: GatedScaleNet (our own)

In addition to `simple` and `resnet18`, this project now also includes a
from-scratch architecture: multi-scale gated blocks (Inception-style
branches + SKNet-style per-channel gating), proper downsampling between
stages, and deep supervision (training-only auxiliary heads) instead of
runtime early-exit. ~3.3M parameters. See `catsdogs/gated_net.py` for the
full design notes.

```bash
python train.py --arch gatedscalenet --epochs 60 --img-size 224 --lr 1e-3
```

Honest expectation: unlike `resnet18 --pretrained`, this architecture has
no pretrained weights (nobody has trained it on a large dataset yet), so
it's learning entirely from these 275 training images. Don't be surprised
if it scores below the pretrained ResNet18 baseline — report whatever
number it actually gets.

