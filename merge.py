import os
def merge_checkpoint(parts_prefix="weights/checkpoint.pth.part", output_path="weights/checkpoint.pth"):
    with open(output_path, 'wb') as output_file:
        i = 0
        while True:
            part_file = f"{parts_prefix}{i}"
            if not os.path.exists(part_file):
                break
            with open(part_file, 'rb') as pf:
                output_file.write(pf.read())
            i += 1
    print(f"Merged {i} parts into {output_path}")
merge_checkpoint()