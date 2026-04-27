from datasets import load_dataset


def load_medqa(split="test"):
    """
    MedQA: USMLE-style medical board exam questions
    Used as secondary evaluation dataset
    """
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")
    return dataset[split]


def format_medqa_case(item):
    """Format MedQA item into patient case text"""
    return f"""CLINICAL QUESTION:
{item['question']}""".strip()


if __name__ == "__main__":
    dataset = load_medqa()
    print(f"Loaded {len(dataset)} MedQA cases")
    print("\nSample:")
    print(format_medqa_case(dataset[0]))
