import os
import os.path
import click
from PIL import Image, ImageFile, UnidentifiedImageError
from PIL.Image import Image as PILImage
from . import interrogate
import hydrus_api
from io import BytesIO
import json
import time
import logging
from datetime import datetime
from typing import Any, Optional, cast

Image.MAX_IMAGE_PIXELS = None
from hydrus_api import APIError


# -----------------------------
# Retry wrapper
# -----------------------------
def get_file_with_retry(client: hydrus_api.Client, file_hash: str, retries: int = 5, delay: int = 5) -> Any:
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


kaomojis: list[str] = [
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
@click.option("--token", required=True, help="Hydrus API token")
@click.option("--cpu", default=True, help="False to Use GPU instead of CPU")
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
def evaluate_api_batch(hashfile: Optional[str], search_tag: tuple[str, ...], token: str, cpu: bool,
                       model: str, threshold: float, host: str,
                       tag_service: str, ratings_only: bool, privacy: bool) -> None:

    # -----------------------------
    # Setup Logging
    # -----------------------------
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"logs/{model}_{timestamp}.log"

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    logging.info("=== WD HYDRUS TAGGER START ===")
    logging.info(f"Model: {model}")
    logging.info(f"CPU Mode: {cpu}")
    logging.info(f"Tag Service: {tag_service}")

    # -----------------------------
    # Load model info
    # -----------------------------
    if not os.path.isfile('./model/' + model + '/info.json'):
        raise ValueError("info.json not found in model folder!")

    with open('./model/' + model + '/info.json') as json_f:
        modelinfo: dict[str, Any] = json.load(json_f)
        done_hashes_file: str = f"done-hashes-{modelinfo['modelname']}.txt"

    if ratings_only and not modelinfo['ratingsflag']:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    interrogator: interrogate.WaifuDiffusionInterrogator = interrogate.WaifuDiffusionInterrogator(
        modelinfo['modelname'], # the name of the model for display purposes
        modelinfo['modelfile'], # the filename of the model file
        modelinfo['tagsfile'], # the filename of the tags file
        model, # the folder storing the previous two files as well as the info file
		modelinfo['ratingsflag'], # flag indicating whether model identifies content rating
		modelinfo['numberofratings'], # amount of tags to consider for content rating if so
        repo_id=modelinfo['source'], # source of the model, credit where credit is due
    )
    interrogator.load(cpu)

    client: hydrus_api.Client = hydrus_api.Client(token, host)

    # -----------------------------
    # Determine file source
    # -----------------------------
    using_tag_search: bool = False

    if search_tag:
        using_tag_search = True
        click.echo(f"Searching Hydrus for tags: {search_tag}")
        logging.info(f"Search Tags: {search_tag}")


        # Build a tag list: search tags + system predicate exclusions
        query_tags: list[str] = list(search_tag) + ["system:filetype is not ugoira, video"]
        client.search_files(tags=query_tags)  # warm up the search to avoid first-query lag
        
        hashes: list[str] = cast(list[str], client.search_files(tags=query_tags, return_hashes=True))


        click.echo(f"Found {len(hashes)} files (excluding ugoira and video).")
        logging.info(f"Found {len(hashes)} files via tag search")

    elif hashfile:
        if not os.path.isfile(hashfile):
            raise ValueError("hashfile not found!")

        with open(hashfile) as f:
            hashes: list[str] = [line.strip() for line in f if line.strip()]

        logging.info(f"Loaded {len(hashes)} hashes from file")

    else:
        raise ValueError("Provide either --search-tag or --hashfile")

    # -----------------------------
    # Done hash tracking (ONLY for hashfile mode)
    # -----------------------------
    done_hashes: set[str] = set()
    bad_hashes: set[str] = set()

    if not using_tag_search:
        if os.path.exists(done_hashes_file):
            with open(done_hashes_file, "r", encoding="utf-8") as f:
                done_hashes = {line.strip() for line in f if line.strip()}

    if os.path.exists("bad-hashes.txt"):
        with open("bad-hashes.txt", "r", encoding="utf-8") as f:
            bad_hashes = {line.strip() for line in f if line.strip()}

    # -----------------------------
    # Processing Loop
    # -----------------------------
    processed_count = 0
    interrupted = False

    try:
        with click.progressbar(hashes) as bar:
            for file_hash in bar:

                file_hash = str(file_hash)

                if not file_hash or file_hash in bad_hashes:
                    continue

                if not using_tag_search and file_hash in done_hashes:
                    continue

                click.echo(" processing: " + file_hash)
                logging.info(f"Processing: {file_hash}")

                try:
                    response: Any = get_file_with_retry(client, file_hash)

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
                image_bytes: BytesIO = BytesIO(response.content)

                try:
                    image: PILImage = Image.open(image_bytes)
                    image = image.convert("RGB")
                except (UnidentifiedImageError, OSError):
                    click.echo(f"Skipping unreadable or non-image file: {file_hash}")
                    logging.warning(f"Skipping unreadable or non-image file: {file_hash}")
                    bad_hashes.add(file_hash)
                    with open("bad-hashes.txt", "a", encoding="utf-8") as bad_f:
                        bad_f.write(file_hash + "\n")
                    continue

                ratings: dict[str, float]
                tags: dict[str, float]
                ratings, tags = interrogator.interrogate(image)

                rating: str = "none"
                if modelinfo['ratingsflag']:
                    ratings["none"] = 0.0
                    for key in ratings.keys():
                        if ratings[key] > ratings[rating]:
                            rating = key

                clipped_tags: list[str] = []

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

                processed_count += 1

                # Only write done-hash if using hashfile mode
                if not using_tag_search:
                    done_hashes.add(file_hash)
                    with open(done_hashes_file, "a", encoding="utf-8") as done_f:
                        done_f.write(file_hash + "\n")

                # --- Micro pause: let tag write commit ---
                time.sleep(0.1)
                # --- Macro pause: only every 200 files ---
                if processed_count % 200 == 0:
                    time.sleep(3)
                    logging.info(f"Processed {processed_count} files")

    except KeyboardInterrupt:
        interrupted = True
        click.echo("\n⚠ Interrupted by user (Ctrl+C)")
        logging.warning("=== INTERRUPTED BY USER (CTRL+C) ===")

    finally:
        logging.info("=== WD HYDRUS TAGGER FINISHED ===")
        logging.info(f"Total processed: {processed_count}")

        if interrupted:
            logging.info("Run ended due to manual interruption.")
        else:
            logging.info("Run completed normally.")

        logging.shutdown()

        click.echo(f"\nFinished. Total processed: {processed_count}")
        
    


if __name__ == '__main__':
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli.add_command(evaluate_api_batch)
    cli()
