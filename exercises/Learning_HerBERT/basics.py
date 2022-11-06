from transformers import AutoTokenizer, AutoModel

# Example code from GitHub

model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1",
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased",
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased",
        "model": "allegro/herbert-large-cased",
    },
}

tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
model = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])

output = model(
    **tokenizer.batch_encode_plus(
        ['Uzależnienie'
],
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
    )
)

print(len(output.last_hidden_state))
print(type(output.last_hidden_state))




