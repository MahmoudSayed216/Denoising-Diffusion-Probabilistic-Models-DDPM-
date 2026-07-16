import torch.nn as nn
import torch




class Embedder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, image_side_length):
        super().__init__()
        self.image_side_length = image_side_length
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.projector = nn.Linear(embedding_dim, image_side_length*image_side_length)


    def forward(self, class_id: int):
        emb = self.embedding_layer(class_id)
        emb = self.projector(emb)
        emb = emb.view((self.image_side_length, self.image_side_length))



class SinudoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, n_timestamps):
        super().__init__()
        numbers = torch.arange(start=0, end=n_timestamps)
        



class ConditionalDDPM(nn.Module):
    def __init__(self, n_timestamps, num_classes, image_side_length, embedding_dim):
        super().__init__()

        self.class_embedder = Embedder(num_classes, embedding_dim, image_side_length)


