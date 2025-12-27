# src/make_dpo.py
import os
import json
import time
from pathlib import Path
import re

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/
BASE = ROOT / "data" / "synthetic_data_generation"

SPEC_PATH    = BASE / "sft_generation_spec.json"

OUT_TRAIN = BASE / "dpo_train.jsonl"
OUT_VAL   = BASE / "dpo_val.jsonl"

TRIES = 1
SEED = 1337
MAX_OUTPUT_TOKENS = 128

N_TRAIN = 500
N_VAL = 100

# TODO: Expand this list by a looooooooooooot
ADJECTIVES = [
    "brave", "shy", "curious", "kind", "sleepy", "cheerful", "gentle", "quiet",
    "tiny", "helpful", "careful", "honest", "silly", "proud", "patient",
]

# TODO: Expand this list by a looooooooooooot
NOUNS = [
    "teapot", "dragon", "rabbit", "robot", "pencil", "cloud", "turtle", "cat",
    "dog", "bear", "kite", "book", "lamp", "cookie", "train",
]

# OOD sets
ADJECTIVES_OOD = [
    "adventurous","agreeable","alert","ancient","angry","anxious","astonished",
    "bouncy","breezy","bright","brisk","bubbly","calm","carefree","charming","chilly",
    "clever","clumsy","colorful","comfy","crafty","cranky","daring","delicate","eager",
    "electric","elegant","enchanted","energetic","fancy","fearless","fidgety","fluffy",
    "fluttery","foggy","fragile","fragrant","friendly","frosty","funny","fuzzy",
    "gentlehearted","giggly","glamorous","glittery","glossy","graceful","grumpy","hasty",
    "hungry","icy","imaginative","jolly","jumpy","lazy","lively","lucky","magical",
    "mellow","mischievous","misty","mysterious","neat","nervous","noisy","odd","optimistic",
    "peaceful","playful","plucky","polite","puzzled","quick","quirky","radiant","restless",
    "roaring","roomy","rosy","rusty","scared","shimmering","shiny","sneezy","sparkly",
    "speedy","spiky","spooky","spotless","springy","stormy","sunlit","swift","tasty",
    "teeny","thirsty","thoughtful","thunderous","tidy","twinkly","wacky","warmhearted",
    "whispery","wiggly","windy","wise","wobbly","zany"
]

NOUNS_OOD = [
    "accordion","acorn","airplane","alarmclock","alligator","anchor","ant","antelope",
    "apple","armadillo","asteroid","astronaut","avocado","backpack","badger","balloon",
    "banana","barn","beehive","bicycle","binoculars","blender","boomerang","broom",
    "bubble","buffalo","butterfly","cactus","calculator","camel","candle","canoe",
    "capybara","caravan","carrot","castle","caterpillar","chameleon","cheesecake","chessboard",
    "chimney","chipmunk","clock","coconut","comet","compass","computer","cookiejar",
    "cornfield","crayon","crocodile","cupcake","dandelion","dinosaur","dolphin","donut",
    "doorbell","dragonfly","drum","eagle","earthworm","eel","elevator","envelope",
    "espresso","fairy","feather","fence","firefly","firetruck","flamingo","flashlight",
    "frog","garden","giraffe","globe","goose","grapefruit","guitar","hamster",
    "harmonica","hedgehog","helmet","honeybee","horseshoe","hotairballoon","icecream",
    "iguana","island","jellyfish","journal","kangaroo","kayak","keyhole","koala",
    "ladder","lantern","lemonade","lighthouse","lollipop","magnet","mango","marble",
    "meadow","microscope","milkshake","mirror","moonbeam","motorboat","mushroom","narwhal",
    "notebook","octopus","omelet","origami","ostrich","paintbrush","pancake","parachute",
    "parrot","peacock","peanut","penguin","peppermint","piano","pinecone","pirateship",
    "popsicle","porcupine","postcard","pumpkin","puzzle","rainbow","raccoon","raindrop",
    "rattle","reindeer","rocket","seahorse","seashell","skateboard","skunk","sloth",
    "snowflake","spaceship","sponge","squirrel","strawberry","submarine","suitcase","sunflower",
    "telescope","thermometer","toaster","tomato","tornado","trampoline","treasurechest","triceratops",
    "trophy","trombone","turtlepond","typewriter","unicorn","vacuum","violin","volcano",
    "waffle","walrus","waterfall","windmill","window","wrench","yacht","yogurt","zebra","zeppelin"
]


api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

with SPEC_PATH.open("r", encoding="utf-8") as f:
    spec = json.load(f)

model = str(spec["model"])
system_prompt = str(spec["system_prompt"])

client = OpenAI(api_key=api_key)
rng = np.random.default_rng(SEED)


def has(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()

def write_split(want_n: int, out_path: Path, split_name: str):

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as f_out:
        id = 0
        while written < want_n:
            print(f"Started Prompt {written}...")

            correct_protagonist = f"{rng.choice(ADJECTIVES_OOD)} {rng.choice(NOUNS_OOD)}"
            prompt_correct = (        
                "Write a very short children's story.\n\n"
                f"- Protagonist: a {correct_protagonist}\n"
                "Style:\n"
                "- Simple words and short sentences.\n"
            )

            wrong_protagonist = correct_protagonist
            while wrong_protagonist.lower() == correct_protagonist.lower():
                wrong_protagonist = f"{rng.choice(ADJECTIVES_OOD)} {rng.choice(NOUNS_OOD)}"

            prompt_wrong = (        
                "Write a very short children's story.\n\n"
                f"- Protagonist: a {wrong_protagonist}\n"
                "Style:\n"
                "- Simple words and short sentences.\n"
            )

            chosen = None

            for attempt in range(TRIES):
                try:
                    r = client.responses.create(
                        model=model,
                        instructions=system_prompt,
                        input=prompt_correct,
                        temperature=0.4,
                        top_p=1.0,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    )
                    txt = (r.output_text or "").strip()

                except Exception:
                    time.sleep(0.25)
                    print("Exception while getting the response for correct prompt..")
                    print(f"Attempt {attempt+1} has failed")
                    continue

                if not has(txt, correct_protagonist):
                    print(f"Correct prompt doesnt have the correct protagonist ({correct_protagonist}), retrying.")
                    continue

                chosen = txt
                break

            if chosen is None:
                print("Chosen is still None --> Rejected. Trying a new prompt.")
                continue

            rejected = None

            for attempt in range(TRIES):
                try:
                    r = client.responses.create(
                        model=model,
                        instructions=system_prompt,
                        input=prompt_wrong,
                        temperature=0.4,
                        top_p=1.0,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    )
                    txt = (r.output_text or "").strip()

                except Exception:
                    time.sleep(0.25)
                    print("Exception while getting the response for wrong prompt..")
                    print(f"Attempt {attempt+1} has failed")
                    continue

                if has(txt, correct_protagonist):
                    print("Wrong prompt has the right protagonist. NOOO. Retrying.")
                    continue

                rejected = txt
                break

            if rejected is None:
                print("Rejected is still None --> Rejected. Trying a new prompt.")
                continue

            f_out.write(json.dumps({
                "id": f"p{id:05d}",
                "split": split_name,
                "prompt": prompt_correct,
                "chosen": chosen,
                "rejected": rejected,
                "violation": "wrong_protagonist",
                "detail": f"a {wrong_protagonist}",
                "correct_protagonist": f"a {correct_protagonist}",
            }, ensure_ascii=False) + "\n")

            id += 1

            print(f"Finished prompt {written}!")
            if written % 5 == 0:
                dt = time.perf_counter() - t0
                avg = dt / max(1, written)
                rem = want_n - written
                eta_min = (rem * avg) / 60.0
                print(f"{split_name}: {written}/{want_n} | avg_s={avg:.2f} | eta_min={eta_min:.1f}")
            
            written += 1


def main():
    print("Starting Train...")
    write_split(N_TRAIN, OUT_TRAIN, "train")
    print("Starting Val...")
    write_split(N_VAL, OUT_VAL, "val")


if __name__ == "__main__":
    main()