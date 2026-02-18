import os.path
import click
from PIL import Image, ImageFile, UnidentifiedImageError
from . import interrogate
import hydrus_api
from io import BytesIO
import json
import time
import logging

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

Image.MAX_IMAGE_PIXELS = None

from hydrus_api import APIError


def get_file_with_retry(client, file_hash, retries=5, delay=5):
    for attempt in range(retries):
        try:
            return client.get_file(file_hash)

        except APIError as e:
            info = e.response.json()

            if info.get("exception_type") == "FileMissingException":
                raise

            if attempt < retries - 1:
                logging.warning(f"retry {attempt} file: {file_hash}")
                time.sleep(delay)
                continue

            logging.error(f"retry fail file: {file_hash}")
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
@click.option("--hashfile", help="Text file containing Hydrus hashes")
@click.option("--search-tag", multiple=True,
              help="Hydrus tag(s) to search for (can be used multiple times)")
@click.option("--token", help="Hydrus API token", required=True)
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option("--model", default="wd-eva02-large-tagger-v3",
              help="Tagging model to use")
@click.option("--threshold", default=0.35,
              help="Threshold to drop tags below")
@click.option("--host", default="http://127.0.0.1:45869",
              help="The URL for your Hydrus server ")
@click.option("--tag-service", default="ai tags",
              help="The Hydrus tag service to add tags to")
@click.option("--ratings-only", default=False,
              help="Strip all tags except content rating")
@click.option("--privacy", default=True,
              help="Hide tag output from cli")
def evaluate_api_batch(hashfile, search_tag, token, cpu, model,
                       threshold, host, tag_service,
                       ratings_only, privacy):

    if not os.path.isfile('./model/' + model + '/info.json'):
        raise ValueError("info.json not found in model folder!")

    with open('./model/' + model + '/info.json') as json_f:
        modelinfo = json.load(json_f)
        done_hashes_file = f"done-hashes-{modelinfo['modelname']}.txt"

    if ratings_only and not modelinfo['ratingsflag']:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    interrogator = interrogate.WaifuDiffusionInterrogator(
        modelinfo['modelname'], # the name of the model for display purposes
        modelinfo['modelfile'], # the filename of the model file
        modelinfo['tagsfile'], # the filename of the tags file
        model, # the folder storing the previous two files as well as the info file
		modelinfo['ratingsflag'], # flag indicating whether model identifies content rating
		modelinfo['numberofratings'], # amount of tags to consider for content rating if so
        repo_id=modelinfo['source'], # source of the model, credit where credit is due
    )
    interrogator.load(cpu)

    client = hydrus_api.Client(token, host)

    # ----------------------------
    # Determine file source
    # ----------------------------
    if search_tag:
        click.echo(f"Searching Hydrus for tags: {search_tag}")
        hashes = client.search_files(tags=list(search_tag))
        click.echo(f"Found {len(hashes)} files.")
    elif hashfile:
        if not os.path.isfile(hashfile):
            raise ValueError("hashfile not found!")
        with open(hashfile) as f:
            hashes = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Provide either --search-tag or --hashfile")

    done_hashes = set()
    bad_hashes = set()

    if os.path.exists(done_hashes_file):
        with open(done_hashes_file, "r", encoding="utf-8") as f:
            done_hashes = {line.strip() for line in f if line.strip()}

    if os.path.exists("bad-hashes.txt"):
        with open("bad-hashes.txt", "r", encoding="utf-8") as f:
            bad_hashes = {line.strip() for line in f if line.strip()}

    with click.progressbar(hashes) as bar:
        for file_hash in bar:

            if not file_hash or file_hash in done_hashes or file_hash in bad_hashes:
                continue

            click.echo(" processing: " + file_hash)

            try:
                response = get_file_with_retry(client, file_hash)

            except hydrus_api.APIError as e:
                info = e.response.json()
                status = info.get("status_code")
                etype = info.get("exception_type")
                click.echo(f"Hydrus error {status} ({etype}) on {file_hash}")
                logging.warning(f"Hydrus error {status} ({etype}) on {file_hash}")

                if etype == "FileMissingException":
                    click.echo(f"Added to bad-hashes.txt: {file_hash}")
                    bad_hashes.add(file_hash)
                    with open("bad-hashes.txt", "a", encoding="utf-8") as bad_f:
                        bad_f.write(file_hash + "\n")
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
                click.echo(f"Skipping unreadable or non-image file: {file_hash}")
                logging.warning(f"Skipping unreadable or non-image file: {file_hash}")
                bad_hashes.add(file_hash)
                with open("bad-hashes.txt", "a", encoding="utf-8") as bad_f:
                    bad_f.write(file_hash + "\n")
                continue

            ratings, tags = interrogator.interrogate(image)

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
                hashes=[file_hash],
                service_names_to_tags={tag_service: clipped_tags}
            )

            done_hashes.add(file_hash)
            with open(done_hashes_file, "a", encoding="utf-8") as done_f:
                done_f.write(file_hash + "\n")
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
