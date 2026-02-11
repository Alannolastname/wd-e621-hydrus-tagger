from pathlib import Path

hashes_file = Path("hashes.txt")
bad_file = Path("bad-hashes.txt")
out_file = Path("hashes.tmp.txt")

# Find all done-hashes-*.txt files
done_files = sorted(Path(".").glob("done-hashes-*.txt"))

if not done_files:
    print("No done-hashes-*.txt files found.")
    exit(1)

print("=========================================")
print("Select file(s) to apply:")
print("-----------------------------------------")

print("1) BAD HASHES (bad-hashes.txt)\n")

for i, f in enumerate(done_files, start=2):
    print(f"{i}) {f.name}")

print("\n0) ALL OF THE ABOVE")
print("=========================================")

try:
    choice = int(input("Select option: "))
except ValueError:
    print("Invalid selection.")
    exit(1)

use_bad_hashes = False
selected_done_files = []

if choice == 0:
    use_bad_hashes = True
    selected_done_files = done_files

elif choice == 1:
    use_bad_hashes = True

elif 2 <= choice <= len(done_files) + 1:
    selected_done_files = [done_files[choice - 2]]

else:
    print("Invalid selection.")
    exit(1)

print("\n=========================================")
print("Cleaning hashes.txt")
print("-----------------------------------------")
print(f"Source file : {hashes_file}")
print(f"Bad file    : {bad_file}")

for i, f in enumerate(selected_done_files, start=1):
    print(f"Done file {i} : {f.name}")

print("=========================================\n")

# Load done hashes
done_hash_map = {}  # hash -> done index

for idx, f in enumerate(selected_done_files, start=1):
    with open(f, "r", encoding="utf-8") as df:
        for line in df:
            h = line.strip()
            if h:
                done_hash_map[h] = idx

# Load bad hashes ONLY if selected
bad_hashes = set()
if use_bad_hashes and bad_file.exists():
    bad_hashes = {
        line.strip()
        for line in bad_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

remove_hashes = set(done_hash_map.keys()) | bad_hashes

total = kept = removed = 0

with hashes_file.open("r", encoding="utf-8") as src, \
     out_file.open("w", encoding="utf-8") as dst:

    for line in src:
        hash_ = line.strip()
        if not hash_:
            continue

        total += 1

        if hash_ in remove_hashes:
            if hash_ in done_hash_map:
                print(f"[REMOVE:DONE-{done_hash_map[hash_]}]   {hash_}")
            else:
                print(f"[REMOVE:BAD]          {hash_}")
            removed += 1
        else:
            print(f"[KEEP]              {hash_}")
            dst.write(hash_ + "\n")
            kept += 1

print("\n=========================================")
print("Summary")
print("-----------------------------------------")
print(f"Total hashes : {total}")
print(f"Kept        : {kept}")
print(f"Removed     : {removed}")
print("=========================================\n")

out_file.replace(hashes_file)
print("hashes.txt updated successfully.")
