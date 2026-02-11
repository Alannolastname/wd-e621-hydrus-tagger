import os

def remove_duplicates_in_file(file_path):
    seen = set()
    unique_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(line)

    # Rewrite file with unique lines only
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(unique_lines)

    removed_count = len(seen) - len(unique_lines)
    return removed_count


def process_files_one_by_one(file_list):
    for file_path in file_list:
        print(f"\nProcessing: {file_path}")

        if not os.path.isfile(file_path):
            print("  ❌ File not found")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        remove_duplicates_in_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            new_lines = f.readlines()

        removed = len(original_lines) - len(new_lines)

        if removed > 0:
            print(f"  🧹 Removed {removed} duplicate lines")
        else:
            print("  ✅ No duplicates found")


# -------- USAGE --------
files_to_process = [
    "hashes.txt",
    "bad-hashes.txt",
    "done-hashes-WD EVA02-Large v3 Tagger.txt",
    "done-hashes-WD V1.4 Vit V2 Tagger.txt",
    "done-hashes-Z3D E621 Convnext Tagger.txt",
]

process_files_one_by_one(files_to_process)
