import os.path

import click
from PIL import Image, ImageFile, UnidentifiedImageError
from . import interrogate
import hydrus_api
from io import BytesIO
import json

import time

import logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

Image.MAX_IMAGE_PIXELS = None



from hydrus_api import APIError

def get_file_with_retry(client, hash, retries=5, delay=5):
    for attempt in range(retries):
        try:
            return client.get_file(hash)

        except APIError as e:
            info = e.response.json()

            if info.get("exception_type") == "FileMissingException":
                # real missing file → skip permanently
                raise

            # otherwise: transient API issue
            if attempt < retries - 1:
                logging.warning(f"retry {attempt} file: {hash}")
                time.sleep(delay)
                continue
            logging.error(f"retry fail file: {hash}")
            raise




kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

@click.group()
def cli():
    pass


@click.command()
@click.argument("hashfile")
@click.option("--token", help="The API token for your Hydrus server")
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option("--model", default="wd-v1-4-vit-tagger-v2", help="The tagging model to use")
@click.option("--threshold", default=0.35, help="The threshhold to drop tags below")
@click.option("--host", default="http://127.0.0.1:45869", help="The URL for your Hydrus server ")
@click.option("--tag-service", default="A.I. Tags", help="The Hydrus tag service to add tags to")
@click.option("--ratings-only", default=False, help="Strip all tags except for content rating")
@click.option("--privacy", default=True, help="hides the tag output from the cli")
def evaluate_api_batch(hashfile, token, cpu, model, threshold, host, tag_service, ratings_only, privacy):
    if not os.path.isfile(hashfile):
        raise ValueError("hashfile not found!")
    if not os.path.isfile('./model/' + model + '/info.json'):
        raise ValueError("info.json not found in model folder!")

    with open('./model/' + model + '/info.json') as json_f:
        modelinfo = json.load(json_f)
        done_hashes_file = f"done-hashes-{modelinfo['modelname']}.txt"


    if ratings_only and not modelinfo['ratingsflag']:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo['modelname'], # the name of the model for display purposes
        modelinfo['modelfile'], # the filename of the model file
        modelinfo['tagsfile'], # the filename of the tags file
        model, # the folder storing the previous two files as well as the info file
		modelinfo['ratingsflag'], # flag indicating whether model identifies content rating
		modelinfo['numberofratings'], # amount of tags to consider for content rating if so
        repo_id=modelinfo['source'], # source of the model, credit where credit is due
    )
    integerator.load(cpu)
    client = hydrus_api.Client(token, host)
    with open(hashfile) as hashfile_f:
        hashes = hashfile_f.readlines()

    done_hashes = set()
    bad_hashes = set()

    if os.path.exists(done_hashes_file):
        with open(done_hashes_file, "r", encoding="utf-8") as f:
            done_hashes = {line.strip() for line in f if line.strip()}


    if os.path.exists("bad-hashes.txt"):
        with open("bad-hashes.txt", "r", encoding="utf-8") as f:
            bad_hashes = {line.strip() for line in f if line.strip()}

    with click.progressbar(hashes) as bar:
        for hash in bar:
            hash = hash.strip()
            if not hash or hash in done_hashes or hash in bad_hashes:
                continue

            click.echo(" processing: " + hash)  

            try:
                response = get_file_with_retry(client, hash)
            
            except hydrus_api.APIError as e:
                info = e.response.json()
                status = info.get("status_code")
                etype = info.get("exception_type")

                click.echo(f"Hydrus error {status} ({etype}) on {hash}")
                logging.warning(f"Hydrus error {status} ({etype}) on {hash}")

                if etype == "FileMissingException":
                    click.echo(f"Added to bad-hashes.txt: {hash}")
                    bad_hashes.add(hash)
                    with open("bad-hashes.txt", "a", encoding="utf-8") as bad_f:
                        bad_f.write(hash + "\n")
                    continue
                else:
                    # real error, not a bad hash
                    raise


            except RuntimeError as e:
                click.echo(str(e))
                click.echo("Hydrus seems unreachable — stopping batch.")
                logging.error(str(e))
                logging.error("Hydrus seems unreachable — stopping batch.")
                raise

            image_bytes = BytesIO(response.content)

            try:
                image = Image.open(image_bytes)
                image = image.convert("RGB")
            except (UnidentifiedImageError, OSError):
                click.echo(f"Skipping unreadable or non-image file: {hash}")
                logging.warning(f"Skipping unreadable or non-image file: {hash}")
                bad_hashes.add(hash)
                with open("bad-hashes.txt", "a", encoding="utf-8") as bad_f:
                    bad_f.write(hash + "\n")
                continue

            ratings, tags = integerator.interrogate(image)  

            rating = "none"
            if modelinfo['ratingsflag']:
                ratings["none"] = 0.0
                for key in ratings.keys():
                    if ratings[key] > ratings[rating]:
                        rating = key    

            clipped_tags = []   

            if not ratings_only:
                for key in tags.keys():
                    if tags[key] > threshold:
                        clipped_tags.append(
                            key.replace("_", " ") if key not in kaomojis else key
                        )   

            if not privacy:
                click.echo("rating: " + rating)
                click.echo("tags: " + ", ".join(clipped_tags))
                click.echo()    

            if modelinfo['ratingsflag']:
                clipped_tags.append("rating:" + rating) 

            if ratings_only:
                clipped_tags.append(
                    "ratings only " + modelinfo['modelname'] + " ai generated tags"
                )
            else:
                clipped_tags.append(
                    modelinfo['modelname'] + " ai generated tags"
                )   

            client.add_tags(
                hashes=[hash],
                service_names_to_tags={tag_service: clipped_tags}
            )   

            done_hashes.add(hash)
            with open(done_hashes_file, "a", encoding="utf-8") as done_f:
                done_f.write(hash + "\n")
                
            # --- Micro pause: let tag write commit ---
            time.sleep(0.1)
            # --- Macro pause: only every 200 files ---
            if len(done_hashes) % 200 == 0:
                time.sleep(3)
                logging.info(f"Processed {len(done_hashes)} files")



if __name__ == '__main__':
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli.add_command(evaluate_api_batch)
    cli()
