# Chunking Strategy

- **Strategy**: Semantic Sliding Window with overlap.
- **Window Size**: 350 tokens with 60 token sliding stride.
- **Entity Preservation**: Prevents splitting Goan Portuguese/Konkani proper nouns (e.g. *Xitt Codi*, *Se Cathedral*, *Reis Magos*).
