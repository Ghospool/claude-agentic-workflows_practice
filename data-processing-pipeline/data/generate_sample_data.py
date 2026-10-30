"""Deterministic synthetic dataset with ground-truth labels, incl. hard cases (sarcasm, negation)."""
import json
import pathlib
import random

random.seed(7)
POS = {"pricing": "The pricing is fair and the value is good.", "quality": "Excellent build quality.",
       "delivery": "Delivery was fast and it arrived early.", "support": "Support was helpful and friendly.",
       "usability": "Very easy to use and intuitive.", "reliability": "Dependable and solid, no issues at all."}
NEG = {"pricing": "Way too expensive for what you get.", "quality": "The quality is poor and it feels cheap.",
       "delivery": "Delivery was late and the package was damaged.", "support": "Support was useless and never replied.",
       "usability": "Confusing interface and hard to use.", "reliability": "Keeps crashing with constant outages."}
NEUTRAL = ["It is okay, nothing special.", "Average product, as described.", "It does the job, okay overall."]
HARD = [("Great, another outage on launch day.", "negative", ["reliability"]),
        ("Not bad at all, quite easy to use.", "positive", ["usability"]),
        ("Love how fast the support ignored me.", "negative", ["support"]),
        ("Oh good, the package arrived late again.", "negative", ["delivery"])]
OPENERS = ["", "Update: ", "Honestly? ", "Review: "]
EMAILS = ["Contact me at jane.doe@example.com.", "Call 415-555-0134 for details."]
rows, labels = [], []
for i in range(1, 301):
    r = random.random()
    if r < 0.07:
        text, sent, topics = random.choice(HARD)
    elif r < 0.17:
        text, sent, topics = random.choice(NEUTRAL), "neutral", ["other"]
    else:
        sent = "positive" if r < 0.55 else "negative"
        ts = random.sample(list(POS), random.choice([1, 1, 2]))
        text, topics = " ".join((POS if sent == "positive" else NEG)[t] for t in ts), ts
    text = random.choice(OPENERS) + text
    if random.random() < 0.05:
        text += " " + random.choice(EMAILS)
    rid = f"R{i:04d}"
    rows.append({"id": rid, "text": text}); labels.append({"id": rid, "sentiment": sent, "topics": sorted(topics)})
here = pathlib.Path(__file__).parent
(here / "reviews.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
(here / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels) + "\n")
print(f"wrote {len(rows)} records")
