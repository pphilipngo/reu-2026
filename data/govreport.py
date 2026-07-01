from datasets import load_dataset

def load_govreport():
    ds = load_dataset('ccdv/govreport-summarization', split='train', streaming=True)
    return iter(ds)

def output_text_and_ref(ds):
    text = next(ds)
    orig_text, reference = text["report"], text["summary"]
    return orig_text, reference, ds

def write_ref(reference, index):
    system_file_name = f"summaries/system_summaries/ref_gov_doc_{index}"
    with open(system_file_name, "w", encoding="utf-8") as file:
        file.write(reference)

ds = load_govreport()

def main():
    for i in range(1, 4):
        text = next(ds)
        orig_text, reference = text["report"], text["summary"]

        system_file_name = f"summaries/system_summaries/ref_gov_doc_{i}"
        with open(system_file_name, "w", encoding="utf-8") as file:
            file.write(reference)

if __name__ == "__main__":
    main()