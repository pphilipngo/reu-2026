from datasets import load_dataset

dataset = load_dataset('ccdv/govreport-summarization', split='train', streaming=True)

it_dataset = iter(dataset)
gov_doc = next(it_dataset)
gov_doc = next(it_dataset)
gov_doc = next(it_dataset)["report"]

def load_govreport(dataset=dataset):
    return str(next(iter(dataset)))


def main():
    for i in range(1, 4):
        text = next(iter(dataset))
        orig_text, reference = text["report"], text["summary"]

        system_file_name = f"summaries/system_summaries/sys_gov_doc_{i}"
        with open(system_file_name, "w", encoding="utf-8") as file:
            file.write(reference)

if __name__ == "__main__":
    main()