import json as json
import torch

device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')

#read vocabulary
with open('Data/vocabulary.json', 'r') as f:
    vocabulary = json.load(f)
    

parameters = {'vocab_size' : len(vocabulary),
	      'embedding_dim':  300,
				'max_length'     :512//2,
				'batch_size'     :16,
				'num_epochs'     :8,
				'random_state'   :123,
				'test_size'      :0.2,
        'device'         : device
}

