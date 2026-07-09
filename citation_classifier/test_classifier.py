import argparse
import pandas as pd

from fastai.text.all import load_learner

MODEL_PATH = "citation_classifier.pkl"

learn = load_learner(MODEL_PATH)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    type=str,
    help="Optional CSV file containing a 'text' column."
)
args = parser.parse_args()

if args.csv:
    df = pd.read_csv(args.csv)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    sentences = df["text"].tolist()

else:
    sentences = [
        "The Rambam writes in Hilchos Teshuvah that repentance requires confession.",
        "Today we are discussing the Rambam's approach to repentance.",
        "The Gemara says, Kol hamosif gorea.",
        "This is a difficult concept to understand.",
        "As the Sfas Emes notes on this parsha, the repetition is deliberate.",
        "The Maharal develops this idea at length in Netzach Yisrael.",
        "The Midrash already raises this question before offering an answer.",
        "This distinction is made explicitly by the Rashba in one of his responsa.",
        "The Baal HaTurim comments that the extra letter is significant.",
        "The Ran understands the Gemara differently from Rashi.",
        "This approach is already implied by the Ramban at the end of the parsha.",
        "The Yerushalmi presents an entirely different perspective on this issue.",
        "As we learn in Pirkei Avos, the reward is according to the effort.",
        "The Sefer HaChinuch explains the underlying purpose of this mitzvah.",
        "The Ohr HaChaim points out that every word in this verse is necessary.",
        "The Gemara in Kiddushin derives this halacha from the wording of the pasuk.",
        "The Tosafos reject Rashi's explanation for two separate reasons.",
        "The Meiri writes that this principle applies even outside the Beis HaMikdash.",
        "This idea appears earlier in the Zohar as well.",
        "The Vilna Gaon explains that the order of the pesukim is intentional.",
        "The Kli Yakar understands the symbolism differently than the Ramban.",
        "The Mishnah in Avos teaches that jealousy removes a person from the world.",
        "The Chasam Sofer explains why this argument was necessary.",
        "The Ibn Ezra offers a completely different reading of the verse.",
        "We should all try to become more patient in our daily lives.",
        "This question has occupied thinkers for generations.",
        "There are several possible ways to understand this idea.",
        "Many people overlook the importance of this lesson.",
        "This approach helps explain why the story unfolds the way it does.",
        "We discussed a similar concept a few weeks ago.",
        "The central message here is one of personal responsibility.",
        "At first glance the passage seems difficult to understand.",
        "This teaches us an important lesson about humility.",
        "There is a fascinating connection between these two topics.",
        "We often fail to notice the deeper meaning of familiar ideas.",
        "The answer becomes much clearer once we look at the broader picture.",
        "Many commentators struggled with this question.",
        "This perspective encourages us to think more carefully about our assumptions.",
        "The Torah emphasizes the importance of kindness throughout our lives."
    ]

for sentence in sentences:
    prediction, index, probabilities = learn.predict(sentence)

    confidence = probabilities[int(index)]

    print("----------------")
    print(sentence)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")