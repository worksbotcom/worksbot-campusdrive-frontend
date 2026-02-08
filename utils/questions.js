export const questions = {
  // "Generative AI": [
  //   {
  //     id: "ga1",
  //     question:
  //       "In a Transformer architecture, what is the primary function of Positional Encoding?",
  //     options: [
  //       "To reduce the computational cost of the self-attention mechanism.",
  //       "To inject information about the relative or absolute position of the tokens in the sequence, as the self-attention mechanism itself is permutation-invariant.",
  //       "To normalize the input embeddings before they are fed into the encoder stack.",
  //       "To increase the dimensionality of the token embeddings to capture more semantic information.",
  //     ],
  //     correctAnswer:
  //       "To inject information about the relative or absolute position of the tokens in the sequence, as the self-attention mechanism itself is permutation-invariant.",
  //   },
  //   {
  //     id: "apt1",
  //     question:
  //       "A number is increased by 25% and then decreased by 20%. The result is what percentage of the original number?",
  //     options: ["100%", "105%", "95%", "110%"],
  //     correctAnswer: "100%",
  //   },
  //   {
  //     id: "ga2",
  //     question:
  //       "What is the core principle behind a Variational Autoencoder (VAE) that distinguishes it from a standard Autoencoder?",
  //     options: [
  //       "It uses a generator and a discriminator in an adversarial setup.",
  //       "It encodes the input into a probability distribution (a mean and variance) in the latent space, rather than a single point.",
  //       "It is exclusively used for generating text, while standard autoencoders are for images.",
  //       "It uses convolutional layers, whereas standard autoencoders only use fully connected layers.",
  //     ],
  //     correctAnswer:
  //       "It encodes the input into a probability distribution (a mean and variance) in the latent space, rather than a single point.",
  //   },
  //   {
  //     id: "apt2",
  //     question: "If 'SYSTEM' is coded as 'SYSMET', how will 'NEARER' be coded?",
  //     options: ["AENRER", "NEAREN", "AENRRN", "AENERN"],
  //     correctAnswer: "AENRER",
  //   },
  //   {
  //     id: "ga3",
  //     question:
  //       "In the context of diffusion models, what happens during the 'reverse process'?",
  //     options: [
  //       "The model systematically adds Gaussian noise to a clean image until it becomes pure noise.",
  //       "The model learns to predict the noise that was added in the forward process and gradually subtracts it from a noise tensor to generate a clean image.",
  //       "The model generates an image in a single step from a random latent vector.",
  //       "The model uses a discriminator to distinguish between real and generated images.",
  //     ],
  //     correctAnswer:
  //       "The model learns to predict the noise that was added in the forward process and gradually subtracts it from a noise tensor to generate a clean image.",
  //   },
  //   {
  //     id: "apt3",
  //     question:
  //       "A, B, and C can complete a piece of work in 24, 6, and 12 days respectively. Working together, they will complete the same work in:",
  //     options: ["1/24 day", "7/24 days", "24/7 days", "4 days"],
  //     correctAnswer: "24/7 days",
  //   },
  //   {
  //     id: "ga4",
  //     question:
  //       "What is the primary trade-off when applying quantization (e.g., 8-bit or 4-bit) to a large language model?",
  //     options: [
  //       "It increases training time but significantly improves inference speed.",
  //       "It reduces the model's memory footprint and increases inference speed, at the cost of a potential loss in precision and performance.",
  //       "It makes the model more robust against adversarial attacks but requires more VRAM.",
  //       "It improves the model's ability to handle multi-lingual tasks but weakens its mathematical reasoning.",
  //     ],
  //     correctAnswer:
  //       "It reduces the model's memory footprint and increases inference speed, at the cost of a potential loss in precision and performance.",
  //   },
  //   {
  //     id: "apt4",
  //     question: "Find the next number in the series: 4, 6, 12, 18, 30, 42, ?",
  //     options: ["56", "60", "66", "72"],
  //     correctAnswer: "60",
  //   },
  //   {
  //     id: "ga5",
  //     question:
  //       "How does Retrieval-Augmented Generation (RAG) primarily help in mitigating hallucinations in LLMs?",
  //     options: [
  //       "By fine-tuning the model on a dataset of factual statements.",
  //       "By increasing the model's parameter count to store more world knowledge.",
  //       "By retrieving relevant, up-to-date information from an external knowledge base and providing it as context to the model for generation.",
  //       "By using a more complex decoding strategy like beam search.",
  //     ],
  //     correctAnswer:
  //       "By retrieving relevant, up-to-date information from an external knowledge base and providing it as context to the model for generation.",
  //   },
  //   {
  //     id: "apt5",
  //     question:
  //       "Two trains are moving in opposite directions at 60 km/h and 90 km/h. Their lengths are 1.10 km and 0.9 km respectively. The time taken by the slower train to cross the faster train in seconds is:",
  //     options: ["36", "45", "48", "52"],
  //     correctAnswer: "48",
  //   },
  //   {
  //     id: "ga6",
  //     question:
  //       "In Generative Adversarial Networks (GANs), what does it signify when the discriminator's loss approaches 0.5 (for binary cross-entropy) and remains stable?",
  //     options: [
  //       "The generator has completely failed and is producing random noise.",
  //       "The system is approaching Nash equilibrium, where the generator is producing images so realistic that the discriminator is no better than random chance at telling them apart from real images.",
  //       "The discriminator has become too powerful, a situation known as mode collapse.",
  //       "The training process has diverged and needs to be restarted with a lower learning rate.",
  //     ],
  //     correctAnswer:
  //       "The system is approaching Nash equilibrium, where the generator is producing images so realistic that the discriminator is no better than random chance at telling them apart from real images.",
  //   },
  //   {
  //     id: "apt6",
  //     question:
  //       "A clock is started at noon. By 10 minutes past 5, the hour hand has turned through:",
  //     options: ["145 degrees", "150 degrees", "155 degrees", "160 degrees"],
  //     correctAnswer: "155 degrees",
  //   },
  //   {
  //     id: "ga7",
  //     question:
  //       "What is the key advantage of using a Mixture of Experts (MoE) architecture in a large language model like Mixtral 8x7B?",
  //     options: [
  //       "It ensures that the model's output is always factually correct.",
  //       "It allows for a very large number of total parameters while only activating a fraction of them for any given token, significantly reducing computational cost during inference.",
  //       "It eliminates the need for an attention mechanism, making the model architecture simpler.",
  //       "It trains eight separate models and averages their outputs, which guarantees better performance.",
  //     ],
  //     correctAnswer:
  //       "It allows for a very large number of total parameters while only activating a fraction of them for any given token, significantly reducing computational cost during inference.",
  //   },
  //   {
  //     id: "apt7",
  //     question:
  //       "A father is now three times as old as his son. Five years back, he was four times as old as his son. The age of the son is:",
  //     options: ["12", "15", "18", "20"],
  //     correctAnswer: "15",
  //   },
  //   {
  //     id: "ga8",
  //     question:
  //       "When fine-tuning a model using Low-Rank Adaptation (LoRA), what is actually being trained?",
  //     options: [
  //       "The entire set of original model weights.",
  //       "Only the embedding and normalization layers of the original model.",
  //       "Two small, low-rank matrices that are injected into each layer, while the large pre-trained weights are kept frozen.",
  //       "A separate, smaller 'adapter' model that corrects the output of the main model.",
  //     ],
  //     correctAnswer:
  //       "Two small, low-rank matrices that are injected into each layer, while the large pre-trained weights are kept frozen.",
  //   },
  //   {
  //     id: "apt8",
  //     question:
  //       "A merchant buys an article for Rs. 27 and sells it at a profit of 10% of the selling price. The selling price is:",
  //     options: ["Rs. 29.70", "Rs. 30", "Rs. 32", "Rs. 37"],
  //     correctAnswer: "Rs. 30",
  //   },
  //   {
  //     id: "ga9",
  //     question:
  //       "How does 'Chain-of-Thought' (CoT) prompting improve the reasoning capabilities of large language models?",
  //     options: [
  //       "It forces the model to use an external calculator for mathematical problems.",
  //       "It reduces the context window to prevent the model from getting distracted.",
  //       "By instructing the model to generate intermediate reasoning steps before the final answer, it allows more computation to be allocated to the problem, mimicking a more deliberate thought process.",
  //       "It translates the problem into a formal logic language that is easier for the model to process.",
  //     ],
  //     correctAnswer:
  //       "By instructing the model to generate intermediate reasoning steps before the final answer, it allows more computation to be allocated to the problem, mimicking a more deliberate thought process.",
  //   },
  //   {
  //     id: "apt9",
  //     question:
  //       "In how many different ways can the letters of the word 'CORPORATION' be arranged so that the vowels always come together?",
  //     options: ["810", "1440", "2880", "50400"],
  //     correctAnswer: "50400",
  //   },
  //   {
  //     id: "ga10",
  //     question:
  //       "What is 'perplexity' in the context of language models, and what does a lower perplexity score indicate?",
  //     options: [
  //       "A measure of model size; lower perplexity means a smaller model.",
  //       "A measure of inference speed; lower perplexity means faster generation.",
  //       "A measure of how well a probability model predicts a sample; a lower perplexity indicates the model is more confident and accurate in its predictions.",
  //       "A measure of creativity; lower perplexity means the model is less likely to generate novel text.",
  //     ],
  //     correctAnswer:
  //       "A measure of how well a probability model predicts a sample; a lower perplexity indicates the model is more confident and accurate in its predictions.",
  //   },
  //   {
  //     id: "apt10",
  //     question:
  //       "Statements: All buildings are houses. No house is an apartment. Conclusions: 1. No building is an apartment. 2. All houses are buildings.",
  //     options: [
  //       "Only conclusion 1 follows",
  //       "Only conclusion 2 follows",
  //       "Both 1 and 2 follow",
  //       "Neither 1 nor 2 follows",
  //     ],
  //     correctAnswer: "Only conclusion 1 follows",
  //   },
  //   {
  //     id: "ga11",
  //     question:
  //       "What is the fundamental difference between 'greedy decoding' and 'beam search' for text generation?",
  //     options: [
  //       "Greedy decoding is probabilistic, while beam search is deterministic.",
  //       "Greedy decoding always picks the single most probable next token, while beam search keeps track of several most probable sequences ('beams') at each step.",
  //       "Greedy decoding can only be used for small models, while beam search is for large models.",
  //       "Beam search is faster but less accurate than greedy decoding.",
  //     ],
  //     correctAnswer:
  //       "Greedy decoding always picks the single most probable next token, while beam search keeps track of several most probable sequences ('beams') at each step.",
  //   },
  //   {
  //     id: "apt11",
  //     question:
  //       "Pointing to a photograph, a man said, 'I have no brother or sister, but that man's father is my father's son.' Whose photograph was it?",
  //     options: ["His own", "His son's", "His father's", "His nephew's"],
  //     correctAnswer: "His son's",
  //   },
  //   {
  //     id: "ga12",
  //     question:
  //       "In multi-modal models like CLIP, how is the connection between images and text established during training?",
  //     options: [
  //       "By training an image captioning model and a text-to-image model separately.",
  //       "By using a GAN to generate images from text descriptions.",
  //       "By jointly training an image encoder and a text encoder to map corresponding image-text pairs to similar representations in a shared latent space.",
  //       "By converting images into a sequence of text tokens and using a standard language model.",
  //     ],
  //     correctAnswer:
  //       "By jointly training an image encoder and a text encoder to map corresponding image-text pairs to similar representations in a shared latent space.",
  //   },
  //   {
  //     id: "apt12",
  //     question:
  //       "A man rows to a place 48 km distant and back in 14 hours. He finds that he can row 4 km with the stream in the same time as 3 km against the stream. The rate of the stream is:",
  //     options: ["1 km/hr", "1.5 km/hr", "2 km/hr", "2.5 km/hr"],
  //     correctAnswer: "1 km/hr",
  //   },
  //   {
  //     id: "ga13",
  //     question:
  //       "What is 'mode collapse' in the context of training Generative Adversarial Networks (GANs)?",
  //     options: [
  //       "When the discriminator becomes too weak to provide a useful gradient to the generator.",
  //       "When the generator learns to produce only one or a limited variety of outputs that can fool the discriminator.",
  //       "A situation where both the generator and discriminator loss functions converge to zero.",
  //       "The final, desired state of training where the model has learned the data distribution perfectly.",
  //     ],
  //     correctAnswer:
  //       "When the generator learns to produce only one or a limited variety of outputs that can fool the discriminator.",
  //   },
  //   {
  //     id: "apt13",
  //     question:
  //       "What is the angle between the minute hand and the hour hand of a clock at 3:40?",
  //     options: ["120 degrees", "125 degrees", "130 degrees", "135 degrees"],
  //     correctAnswer: "130 degrees",
  //   },
  //   {
  //     id: "ga14",
  //     question:
  //       "The 'temperature' parameter in language model sampling controls:",
  //     options: [
  //       "The number of beams in a beam search.",
  //       "The randomness of the output. Higher temperatures lead to more random, creative outputs, while lower temperatures make the output more deterministic and focused.",
  //       "The maximum length of the generated sequence.",
  //       "The amount of factual information the model can access from its training data.",
  //     ],
  //     correctAnswer:
  //       "The randomness of the output. Higher temperatures lead to more random, creative outputs, while lower temperatures make the output more deterministic and focused.",
  //   },
  //   {
  //     id: "apt14",
  //     question:
  //       "A bag contains 2 red, 3 green and 2 blue balls. Two balls are drawn at random. What is the probability that none of the balls drawn is blue?",
  //     options: ["10/21", "11/21", "2/7", "5/7"],
  //     correctAnswer: "10/21",
  //   },
  //   {
  //     id: "ga15",
  //     question:
  //       "How does Reinforcement Learning from Human Feedback (RLHF) contribute to the alignment of large language models?",
  //     options: [
  //       "It teaches the model new facts from human-written documents.",
  //       "It fine-tunes the model to optimize for a reward signal learned from human preferences, making it more helpful, harmless, and honest.",
  //       "It uses human feedback to compress the model to a smaller size.",
  //       "It is a pre-training technique used to build the foundational model from scratch.",
  //     ],
  //     correctAnswer:
  //       "It fine-tunes the model to optimize for a reward signal learned from human preferences, making it more helpful, harmless, and honest.",
  //   },
  //   {
  //     id: "apt15",
  //     question:
  //       "The ratio of the ages of two students is 3:2. One is older than the other by 5 years. What is the age of the younger student?",
  //     options: ["2 years", "10 years", "15 years", "2.5 years"],
  //     correctAnswer: "10 years",
  //   },
  //   {
  //     id: "ga16",
  //     question:
  //       "What is the 'curse of dimensionality' and how does it relate to generative models?",
  //     options: [
  //       "It refers to the fact that model performance linearly decreases with the number of dimensions.",
  //       "It describes the phenomenon where the volume of the data space grows so fast that available data becomes sparse, making it difficult for models to learn a meaningful data distribution.",
  //       "It is the challenge of visualizing high-dimensional data.",
  //       "It refers to the maximum number of parameters a model can have before it starts overfitting.",
  //     ],
  //     correctAnswer:
  //       "It describes the phenomenon where the volume of the data space grows so fast that available data becomes sparse, making it difficult for models to learn a meaningful data distribution.",
  //   },
  //   {
  //     id: "apt16",
  //     question:
  //       "A car covers a certain distance at a speed of 60 km/hr in 4 hours. To cover the same distance in 3 hours, it must travel at a speed of:",
  //     options: ["70 km/hr", "75 km/hr", "80 km/hr", "90 km/hr"],
  //     correctAnswer: "80 km/hr",
  //   },
  //   {
  //     id: "ga17",
  //     question:
  //       "Why is the cross-attention mechanism crucial in an encoder-decoder Transformer architecture (like the original one for machine translation)?",
  //     options: [
  //       "It allows the encoder to understand the positions of words in the input sentence.",
  //       "It allows each token in the decoder to 'look at' all the tokens from the encoder's output, helping it to focus on relevant parts of the source sequence when generating the target sequence.",
  //       "It is a regularization technique to prevent overfitting during training.",
  //       "It enables the model to process multiple languages simultaneously.",
  //     ],
  //     correctAnswer:
  //       "It allows each token in the decoder to 'look at' all the tokens from the encoder's output, helping it to focus on relevant parts of the source sequence when generating the target sequence.",
  //   },
  //   {
  //     id: "apt17",
  //     question: "Find the odd one out: 3, 5, 11, 14, 17, 21",
  //     options: ["5", "11", "14", "21"],
  //     correctAnswer: "14",
  //   },
  //   {
  //     id: "ga18",
  //     question:
  //       "What is a significant limitation of autoregressive generative models (like GPT) compared to diffusion models for image generation?",
  //     options: [
  //       "They are much slower to train.",
  //       "They generate images in a fixed, raster-scan order (pixel by pixel), which can lead to less coherent global structures compared to the holistic refinement process of diffusion models.",
  //       "They cannot generate high-resolution images.",
  //       "They require labeled data, whereas diffusion models are unsupervised.",
  //     ],
  //     correctAnswer:
  //       "They generate images in a fixed, raster-scan order (pixel by pixel), which can lead to less coherent global structures compared to the holistic refinement process of diffusion models.",
  //   },
  //   {
  //     id: "apt18",
  //     question:
  //       "The total of the ages of Amar, Akbar and Anthony is 80 years. What was the total of their ages three years ago?",
  //     options: ["71 years", "74 years", "77 years", "72 years"],
  //     correctAnswer: "71 years",
  //   },
  //   {
  //     id: "ga19",
  //     question:
  //       "What problem does 'top-p' (or nucleus) sampling solve that is not addressed by 'temperature' sampling alone?",
  //     options: [
  //       "It prevents the model from ever repeating words.",
  //       "It allows for dynamic selection of the number of choices in the vocabulary, by considering the smallest set of tokens whose cumulative probability is at least 'p', thus avoiding sampling from the long tail of unlikely tokens.",
  //       "It guarantees the generated text will be grammatically correct.",
  //       "It is a method for fine-tuning, not sampling.",
  //     ],
  //     correctAnswer:
  //       "It allows for dynamic selection of the number of choices in the vocabulary, by considering the smallest set of tokens whose cumulative probability is at least 'p', thus avoiding sampling from the long tail of unlikely tokens.",
  //   },
  //   {
  //     id: "apt19",
  //     question:
  //       "A grocer has a sale of Rs. 6435, Rs. 6927, Rs. 6855, Rs. 7230 and Rs. 6562 for 5 consecutive months. How much sale must he have in the sixth month so that he gets an average sale of Rs. 6500?",
  //     options: ["Rs. 4991", "Rs. 5991", "Rs. 6001", "Rs. 6991"],
  //     correctAnswer: "Rs. 4991",
  //   },
  //   {
  //     id: "ga20",
  //     question:
  //       "Which statement best describes the concept of 'emergent abilities' in large language models?",
  //     options: [
  //       "Abilities that are explicitly programmed into the model by its creators.",
  //       "The model's ability to learn new languages without being trained on them.",
  //       "Abilities that are not present in smaller models but appear in larger models, often unpredictably, as the model scales in size and data.",
  //       "The model's ability to generate its own training data.",
  //     ],
  //     correctAnswer:
  //       "Abilities that are not present in smaller models but appear in larger models, often unpredictably, as the model scales in size and data.",
  //   },
  //   {
  //     id: "apt20",
  //     question:
  //       "How many 3-digit numbers can be formed from the digits 2, 3, 5, 6, 7 and 9, which are divisible by 5 and none of the digits is repeated?",
  //     options: ["5", "10", "15", "20"],
  //     correctAnswer: "20",
  //   },
  //   {
  //     id: "ga21",
  //     question:
  //       "What is the primary motivation for using self-supervised learning to pre-train large foundational models?",
  //     options: [
  //       "It is the only method that works for Transformer architectures.",
  //       "It allows models to learn from vast amounts of unlabeled data, which is far more abundant than labeled data, by creating pretext tasks (like predicting a masked word).",
  //       "It guarantees that the model will not exhibit any bias.",
  //       "It is computationally cheaper than supervised learning on a per-sample basis.",
  //     ],
  //     correctAnswer:
  //       "It allows models to learn from vast amounts of unlabeled data, which is far more abundant than labeled data, by creating pretext tasks (like predicting a masked word).",
  //   },
  //   {
  //     id: "apt21",
  //     question:
  //       "A man on a tour travels first 160 km at 64 km/hr and the next 160 km at 80 km/hr. The average speed for the first 320 km of the tour is:",
  //     options: ["35.55 km/hr", "71.11 km/hr", "72 km/hr", "36 km/hr"],
  //     correctAnswer: "71.11 km/hr",
  //   },
  //   {
  //     id: "ga22",
  //     question:
  //       "In the context of AI safety, what does 'bias amplification' refer to?",
  //     options: [
  //       "The process of intentionally adding bias to a model to achieve a desired outcome.",
  //       "A technique for making models more fair and equitable.",
  //       "The phenomenon where a model takes existing biases present in its training data and reinforces or even strengthens them in its outputs.",
  //       "The increase in model performance on a biased test set.",
  //     ],
  //     correctAnswer:
  //       "The phenomenon where a model takes existing biases present in its training data and reinforces or even strengthens them in its outputs.",
  //   },
  //   {
  //     id: "apt22",
  //     question: "The cube root of .000216 is:",
  //     options: ["0.6", "0.06", "0.006", "0.0006"],
  //     correctAnswer: "0.06",
  //   },
  //   {
  //     id: "ga23",
  //     question:
  //       "What is a key difference between StyleGAN and earlier GAN architectures?",
  //     options: [
  //       "StyleGAN operates exclusively on text data.",
  //       "It introduced a mapping network and adaptive instance normalization (AdaIN) to allow for more disentangled and controllable manipulation of styles at different levels of detail (e.g., coarse, medium, fine).",
  //       "It was the first GAN to use a diffusion-based generator.",
  //       "It is a supervised model that requires detailed labels for every training image.",
  //     ],
  //     correctAnswer:
  //       "It introduced a mapping network and adaptive instance normalization (AdaIN) to allow for more disentangled and controllable manipulation of styles at different levels of detail (e.g., coarse, medium, fine).",
  //   },
  //   {
  //     id: "apt23",
  //     question:
  //       "If a person walks at 14 km/hr instead of 10 km/hr, he would have walked 20 km more. The actual distance travelled by him is:",
  //     options: ["50 km", "56 km", "70 km", "80 km"],
  //     correctAnswer: "50 km",
  //   },
  //   {
  //     id: "ga24",
  //     question:
  //       "When an LLM is said to 'hallucinate', what is the most accurate technical description of what is happening?",
  //     options: [
  //       "The model is intentionally deceiving the user.",
  //       "The model is accessing a hidden, creative part of its architecture.",
  //       "The model is generating plausible-sounding but factually incorrect or nonsensical information because it is fundamentally a probabilistic sequence generator, not a knowledge retriever.",
  //       "It's a sign that the model's weights have been corrupted during inference.",
  //     ],
  //     correctAnswer:
  //       "The model is generating plausible-sounding but factually incorrect or nonsensical information because it is fundamentally a probabilistic sequence generator, not a knowledge retriever.",
  //   },
  //   {
  //     id: "apt24",
  //     question:
  //       "A tank is filled by a tap in 4 hours while it is emptied by another tap in 9 hours. If both taps are opened simultaneously, in how many hours will the tank be filled?",
  //     options: ["5.4 hours", "6.8 hours", "7.2 hours", "8.1 hours"],
  //     correctAnswer: "7.2 hours",
  //   },
  //   {
  //     id: "ga25",
  //     question:
  //       "What is the 'attention head' in a multi-head self-attention mechanism?",
  //     options: [
  //       "The final output layer of the Transformer model.",
  //       "A single, complete attention calculation. Using multiple heads allows the model to jointly attend to information from different representation subspaces at different positions.",
  //       "The part of the model that decides which tokens to attend to.",
  //       "A hyperparameter that controls the overall size of the model.",
  //     ],
  //     correctAnswer:
  //       "A single, complete attention calculation. Using multiple heads allows the model to jointly attend to information from different representation subspaces at different positions.",
  //   },
  //   {
  //     id: "apt25",
  //     question:
  //       "A sum of money at simple interest amounts to Rs. 815 in 3 years and to Rs. 854 in 4 years. The sum is:",
  //     options: ["Rs. 650", "Rs. 690", "Rs. 698", "Rs. 700"],
  //     correctAnswer: "Rs. 698",
  //   },
  // ],
  // "Full Stack Python Developer": [
  //   {
  //     id: "fspy1",
  //     question: "In Python, what is the primary purpose of a metaclass?",
  //     options: [
  //       "To create singleton instances of a class.",
  //       "To define the behavior of a class itself, rather than the behavior of its instances.",
  //       "To enforce type hinting at runtime.",
  //       "To automatically generate documentation for classes.",
  //     ],
  //     correctAnswer:
  //       "To define the behavior of a class itself, rather than the behavior of its instances.",
  //   },
  //   {
  //     id: "fspy2",
  //     question:
  //       "When optimizing a Django ORM query involving a many-to-one relationship (ForeignKey), which method is generally more efficient for reducing database hits?",
  //     options: ["prefetch_related()", "select_related()", "defer()", "only()"],
  //     correctAnswer: "select_related()",
  //   },
  //   {
  //     id: "fspy3",
  //     question:
  //       "Consider the JavaScript code: `console.log('1'); setTimeout(() => console.log('2'), 0); Promise.resolve().then(() => console.log('3')); console.log('4');`. What is the output order?",
  //     options: ["1, 2, 3, 4", "1, 4, 2, 3", "1, 4, 3, 2", "1, 3, 4, 2"],
  //     correctAnswer: "1, 4, 3, 2",
  //   },
  //   {
  //     id: "fspy4",
  //     question:
  //       "What is the primary advantage of using a multi-stage build in a Dockerfile for a Python web application?",
  //     options: [
  //       "It speeds up the `docker build` command by caching more layers.",
  //       "It allows the final production image to be significantly smaller by excluding build-time dependencies and compilers.",
  //       "It enables running unit and integration tests inside the same Dockerfile.",
  //       "It provides a way to run multiple application entrypoints from a single image.",
  //     ],
  //     correctAnswer:
  //       "It allows the final production image to be significantly smaller by excluding build-time dependencies and compilers.",
  //   },
  //   {
  //     id: "fspy5",
  //     question:
  //       "In Python's `asyncio`, what is the key difference between `asyncio.gather()` and `asyncio.wait()`?",
  //     options: [
  //       "`gather` is a high-level function that returns an aggregated list of results, while `wait` is a lower-level function that returns sets of done and pending futures.",
  //       "`wait` can only handle futures, while `gather` can handle coroutines directly.",
  //       "`gather` runs tasks sequentially, whereas `wait` runs them concurrently.",
  //       "There is no functional difference; `gather` is just an alias for `wait`.",
  //     ],
  //     correctAnswer:
  //       "`gather` is a high-level function that returns an aggregated list of results, while `wait` is a lower-level function that returns sets of done and pending futures.",
  //   },
  //   {
  //     id: "fspy6",
  //     question:
  //       "In CSS, what is the specificity of the selector `div#main .content p:first-child`?",
  //     options: ["1, 2, 2", "0, 1, 2, 3", "1, 1, 3", "0, 1, 1, 3"],
  //     correctAnswer: "1, 2, 2",
  //   },
  //   {
  //     id: "fspy7",
  //     question:
  //       "What is the role of the Global Interpreter Lock (GIL) in CPython and how does it affect multi-threaded applications?",
  //     options: [
  //       "It prevents race conditions by allowing only one thread to access any Python object at a time.",
  //       "It prevents true parallel execution of Python byte-code in multi-core CPUs by ensuring only one thread executes Python code at a time.",
  //       "It improves performance by locking the entire interpreter to speed up single-threaded code.",
  //       "It is a mechanism for managing global variables across different threads.",
  //     ],
  //     correctAnswer:
  //       "It prevents true parallel execution of Python byte-code in multi-core CPUs by ensuring only one thread executes Python code at a time.",
  //   },
  //   {
  //     id: "fspy8",
  //     question:
  //       "In a PostgreSQL database, what is the main purpose of the `VACUUM` command?",
  //     options: [
  //       "To permanently delete all data from a table.",
  //       "To shrink the physical size of the database files on disk.",
  //       "To reclaim storage occupied by dead tuples and update data statistics for the query planner.",
  //       "To create a backup of the database.",
  //     ],
  //     correctAnswer:
  //       "To reclaim storage occupied by dead tuples and update data statistics for the query planner.",
  //   },
  //   {
  //     id: "fspy9",
  //     question:
  //       "In Flask, what is the difference between the 'application context' and the 'request context'?",
  //     options: [
  //       "They are the same; the terms are interchangeable.",
  //       "The application context is tied to the lifecycle of the application, while the request context is tied to an individual HTTP request.",
  //       "The application context holds configuration, while the request context holds user session data.",
  //       "The request context is available in all threads, but the application context is thread-local.",
  //     ],
  //     correctAnswer:
  //       "The application context is tied to the lifecycle of the application, while the request context is tied to an individual HTTP request.",
  //   },
  //   {
  //     id: "fspy10",
  //     question:
  //       "What is the primary difference between `Promise.all()` and `Promise.allSettled()` in JavaScript?",
  //     options: [
  //       "`all` waits for all promises to be resolved, while `allSettled` waits for just the first one.",
  //       "`all` rejects as soon as one of the promises rejects, while `allSettled` waits for all promises to either resolve or reject and returns an array of their outcomes.",
  //       "`allSettled` is a newer syntax for `all` with no functional difference.",
  //       "`all` returns an array of results, while `allSettled` returns a single value from the first promise to settle.",
  //     ],
  //     correctAnswer:
  //       "`all` rejects as soon as one of the promises rejects, while `allSettled` waits for all promises to either resolve or reject and returns an array of their outcomes.",
  //   },
  //   {
  //     id: "fspy11",
  //     question:
  //       "When implementing database transactions in Django, what does `transaction.atomic()` ensure?",
  //     options: [
  //       "It ensures that a block of code is executed asynchronously to avoid blocking the database.",
  //       "It guarantees that a group of database operations are executed as a single, indivisible unit (ACID compliance).",
  //       "It automatically optimizes the queries within the block.",
  //       "It creates a read-only transaction to prevent data modification.",
  //     ],
  //     correctAnswer:
  //       "It guarantees that a group of database operations are executed as a single, indivisible unit (ACID compliance).",
  //   },
  //   {
  //     id: "fspy12",
  //     question:
  //       "In Python, how does a generator function differ from a regular function that returns a list?",
  //     options: [
  //       "A generator can only be used within a class.",
  //       "A generator uses the `yield` keyword to produce a sequence of values lazily (one at a time), which is more memory-efficient for large datasets.",
  //       "A regular function is faster because it computes all values at once.",
  //       "There is no difference in memory usage, only in syntax.",
  //     ],
  //     correctAnswer:
  //       "A generator uses the `yield` keyword to produce a sequence of values lazily (one at a time), which is more memory-efficient for large datasets.",
  //   },
  //   {
  //     id: "fspy13",
  //     question:
  //       "What is the primary purpose of an API Gateway in a microservices architecture?",
  //     options: [
  //       "To act as a database for storing API schemas.",
  //       "To provide a single entry point for all clients, handling tasks like routing, authentication, rate limiting, and caching.",
  //       "To monitor the uptime of individual microservices.",
  //       "To automatically scale microservices based on traffic.",
  //     ],
  //     correctAnswer:
  //       "To provide a single entry point for all clients, handling tasks like routing, authentication, rate limiting, and caching.",
  //   },
  //   {
  //     id: "fspy14",
  //     question:
  //       "In React, what is the core problem that the `useCallback` hook is designed to solve?",
  //     options: [
  //       "To prevent infinite loops in `useEffect` hooks.",
  //       "To memoize a function definition so it isn't recreated on every render, thus preventing unnecessary re-renders of child components that depend on it.",
  //       "To fetch data asynchronously within a component.",
  //       "To store state that persists across renders without causing a re-render.",
  //     ],
  //     correctAnswer:
  //       "To memoize a function definition so it isn't recreated on every render, thus preventing unnecessary re-renders of child components that depend on it.",
  //   },
  //   {
  //     id: "fspy15",
  //     question:
  //       "Which HTTP header is crucial for a web server to send to a browser to mitigate Cross-Site Scripting (XSS) attacks?",
  //     options: [
  //       "Access-Control-Allow-Origin",
  //       "Content-Security-Policy",
  //       "Strict-Transport-Security",
  //       "Cache-Control",
  //     ],
  //     correctAnswer: "Content-Security-Policy",
  //   },
  //   {
  //     id: "fspy16",
  //     question:
  //       "In the context of database indexing, what is a 'covering index'?",
  //     options: [
  //       "An index that includes all columns of a table.",
  //       "An index that can satisfy a query entirely from the index itself, without having to access the table data.",
  //       "A special type of index used exclusively for full-text search.",
  //       "The primary key of a table.",
  //     ],
  //     correctAnswer:
  //       "An index that can satisfy a query entirely from the index itself, without having to access the table data.",
  //   },
  //   {
  //     id: "fspy17",
  //     question:
  //       "In Django, what is the execution order of middleware for a request versus a response?",
  //     options: [
  //       "Request: top-to-bottom. Response: random.",
  //       "Request: top-to-bottom. Response: top-to-bottom.",
  //       "Request: bottom-to-top. Response: top-to-bottom.",
  //       "Request: top-to-bottom. Response: bottom-to-top.",
  //     ],
  //     correctAnswer: "Request: top-to-bottom. Response: bottom-to-top.",
  //   },
  //   {
  //     id: "fspy18",
  //     question: "What does the `__slots__` attribute in a Python class do?",
  //     options: [
  //       "It defines a list of allowed method names for the class.",
  //       "It provides a more memory-efficient object structure by pre-allocating space for a fixed set of attributes and preventing the creation of `__dict__`.",
  //       "It makes all attributes read-only after they are initialized.",
  //       "It is a mechanism for creating class-level variables.",
  //     ],
  //     correctAnswer:
  //       "It provides a more memory-efficient object structure by pre-allocating space for a fixed set of attributes and preventing the creation of `__dict__`.",
  //   },
  //   {
  //     id: "fspy19",
  //     question:
  //       "In a REST vs. GraphQL comparison, what is a key advantage of GraphQL?",
  //     options: [
  //       "GraphQL is inherently more secure than REST.",
  //       "GraphQL allows clients to request exactly the data they need, preventing over-fetching and under-fetching.",
  //       "GraphQL uses the more efficient TCP protocol while REST is limited to HTTP.",
  //       "GraphQL has built-in caching mechanisms that REST lacks.",
  //     ],
  //     correctAnswer:
  //       "GraphQL allows clients to request exactly the data they need, preventing over-fetching and under-fetching.",
  //   },
  //   {
  //     id: "fspy20",
  //     question:
  //       "How does a WSGI (Web Server Gateway Interface) server like Gunicorn or uWSGI work with a Python web framework like Django or Flask?",
  //     options: [
  //       "It acts as a reverse proxy, forwarding requests directly to the framework's development server.",
  //       "It is a standardized interface that allows a web server to forward requests to a Python web application and receive responses.",
  //       "It compiles the Python code into a faster, executable binary before running it.",
  //       "It is a database adapter for connecting the framework to a SQL database.",
  //     ],
  //     correctAnswer:
  //       "It is a standardized interface that allows a web server to forward requests to a Python web application and receive responses.",
  //   },
  //   {
  //     id: "fspy21",
  //     question:
  //       "What is the primary function of a `context manager` in Python (created with the `with` statement)?",
  //     options: [
  //       "To manage the global state of an application.",
  //       "To ensure that resources are properly acquired and released, even if errors occur (e.g., closing a file or a database connection).",
  //       "To create a separate memory space for a block of code.",
  //       "To handle asynchronous operations without using `async`/`await`.",
  //     ],
  //     correctAnswer:
  //       "To ensure that resources are properly acquired and released, even if errors occur (e.g., closing a file or a database connection).",
  //   },
  //   {
  //     id: "fspy22",
  //     question:
  //       "In React, what is the difference between a controlled component and an uncontrolled component?",
  //     options: [
  //       "Controlled components are for class components, while uncontrolled are for functional components.",
  //       "In a controlled component, form data is handled by the React component's state. In an uncontrolled component, form data is handled by the DOM itself.",
  //       "Controlled components always re-render when their props change, while uncontrolled components do not.",
  //       "Uncontrolled components are less secure and should not be used for forms with sensitive data.",
  //     ],
  //     correctAnswer:
  //       "In a controlled component, form data is handled by the React component's state. In an uncontrolled component, form data is handled by the DOM itself.",
  //   },
  //   {
  //     id: "fspy23",
  //     question:
  //       "In the context of JWT (JSON Web Tokens), what is the purpose of the signature?",
  //     options: [
  //       "To encrypt the payload so it cannot be read.",
  //       "To verify that the sender of the JWT is who it says it is and to ensure that the message wasn't changed along the way.",
  //       "To reduce the size of the token for faster transmission.",
  //       "To store user permissions and roles.",
  //     ],
  //     correctAnswer:
  //       "To verify that the sender of the JWT is who it says it is and to ensure that the message wasn't changed along the way.",
  //   },
  //   {
  //     id: "fspy24",
  //     question:
  //       "What is tree shaking in the context of modern JavaScript bundlers like Webpack or Vite?",
  //     options: [
  //       "A process of re-organizing the component tree for faster rendering in React.",
  //       "A debugging technique for finding memory leaks.",
  //       "A form of dead code elimination that removes unused exports from the final bundle, resulting in a smaller file size.",
  //       "A method for dynamically loading components only when they are needed.",
  //     ],
  //     correctAnswer:
  //       "A form of dead code elimination that removes unused exports from the final bundle, resulting in a smaller file size.",
  //   },
  //   {
  //     id: "fspy25",
  //     question:
  //       "When would you choose to use Redis as a primary database instead of a traditional RDBMS like PostgreSQL?",
  //     options: [
  //       "For applications requiring complex transactions and strict data consistency.",
  //       "For applications requiring extremely low-latency read/write operations on simple key-value data structures.",
  //       "For storing large binary files like images and videos.",
  //       "When you need to perform complex analytical queries with many JOINs.",
  //     ],
  //     correctAnswer:
  //       "For applications requiring extremely low-latency read/write operations on simple key-value data structures.",
  //   },
  //   {
  //     id: "fspy26",
  //     question:
  //       "In a Django project, what is the main purpose of the `AppConfig` class in `apps.py`?",
  //     options: [
  //       "To define the URL patterns for an application.",
  //       "To store application-specific settings that should not be in `settings.py`.",
  //       "To allow for application configuration and to hook into Django's initialization process, for tasks like registering signals.",
  //       "To specify the database models for an application.",
  //     ],
  //     correctAnswer:
  //       "To allow for application configuration and to hook into Django's initialization process, for tasks like registering signals.",
  //   },
  //   {
  //     id: "fspy27",
  //     question:
  //       "According to the CAP theorem, a distributed system can only provide two of three guarantees. What are they?",
  //     options: [
  //       "Concurrency, Atomicity, Performance",
  //       "Consistency, Availability, Partition Tolerance",
  //       "Confidentiality, Integrity, Availability",
  //       "Complexity, Accuracy, Persistence",
  //     ],
  //     correctAnswer: "Consistency, Availability, Partition Tolerance",
  //   },
  //   {
  //     id: "fspy28",
  //     question:
  //       "What is the primary mechanism `pytest` uses to provide resources like database connections or temporary files to test functions?",
  //     options: [
  //       "Global variables",
  //       "Class inheritance",
  //       "Fixtures",
  //       "Monkeypatching",
  //     ],
  //     correctAnswer: "Fixtures",
  //   },
  //   {
  //     id: "fspy29",
  //     question:
  //       "How can you mitigate a CSRF (Cross-Site Request Forgery) attack in a Django application?",
  //     options: [
  //       "By using HTTPS for all connections.",
  //       "By escaping all user-submitted data before rendering it in templates.",
  //       "By enabling Django's `CsrfViewMiddleware` and using the `{% csrf_token %}` template tag in forms.",
  //       "By setting strong passwords for all user accounts.",
  //     ],
  //     correctAnswer:
  //       "By enabling Django's `CsrfViewMiddleware` and using the `{% csrf_token %}` template tag in forms.",
  //   },
  //   {
  //     id: "fspy30",
  //     question:
  //       "In Python's type hinting, what is the difference between `List` and `list`?",
  //     options: [
  //       "`List` is used for runtime type checking, while `list` is only for static analysis.",
  //       "There is no difference; they are interchangeable.",
  //       "Before Python 3.9, `List` (from `typing`) must be used to provide generic type parameters (e.g., `List[int]`), while `list` could not.",
  //       "`list` is an abstract base class, while `List` is a concrete implementation.",
  //     ],
  //     correctAnswer:
  //       "Before Python 3.9, `List` (from `typing`) must be used to provide generic type parameters (e.g., `List[int]`), while `list` could not.",
  //   },
  //   {
  //     id: "fspy31",
  //     question: "What is the 'event delegation' pattern in JavaScript?",
  //     options: [
  //       "A pattern where you assign event listeners to every child element individually.",
  //       "The process of the browser deciding which event to fire first.",
  //       "A technique where you add a single event listener to a parent element to manage events for all of its children, leveraging event bubbling.",
  //       "A way to create custom events.",
  //     ],
  //     correctAnswer:
  //       "A technique where you add a single event listener to a parent element to manage events for all of its children, leveraging event bubbling.",
  //   },
  //   {
  //     id: "fspy32",
  //     question:
  //       "What is the purpose of the `EXPOSE` instruction in a Dockerfile?",
  //     options: [
  //       "It publishes the port to the host machine, making it accessible from the outside.",
  //       "It acts as documentation, indicating which ports the containerized application listens on.",
  //       "It creates a firewall rule to allow traffic on that port.",
  //       "It is required to connect containers on the same network.",
  //     ],
  //     correctAnswer:
  //       "It acts as documentation, indicating which ports the containerized application listens on.",
  //   },
  //   {
  //     id: "fspy33",
  //     question:
  //       "In a SQL database, what is the difference between a `LEFT JOIN` and an `INNER JOIN`?",
  //     options: [
  //       "An `INNER JOIN` returns all rows from both tables, while a `LEFT JOIN` only returns rows from the left table.",
  //       "An `INNER JOIN` returns only the rows where the join condition is met in both tables. A `LEFT JOIN` returns all rows from the left table, and the matched rows from theright table, or NULL if there is no match.",
  //       "A `LEFT JOIN` is more performant than an `INNER JOIN`.",
  //       "They are functionally identical, but `LEFT JOIN` is part of the SQL-92 standard.",
  //     ],
  //     correctAnswer:
  //       "An `INNER JOIN` returns only the rows where the join condition is met in both tables. A `LEFT JOIN` returns all rows from the left table, and the matched rows from theright table, or NULL if there is no match.",
  //   },
  //   {
  //     id: "fspy34",
  //     question:
  //       "What problem does a message queue (like RabbitMQ or Kafka) solve in a distributed system?",
  //     options: [
  //       "It acts as a primary data store for user information.",
  //       "It provides a way to store large files in the cloud.",
  //       "It enables asynchronous communication and decouples services, improving scalability and resilience.",
  //       "It enforces synchronous, request-response communication between services.",
  //     ],
  //     correctAnswer:
  //       "It enables asynchronous communication and decouples services, improving scalability and resilience.",
  //   },
  //   {
  //     id: "fspy35",
  //     question:
  //       "In Django REST Framework, what is the primary difference between a `Serializer` and a `ModelSerializer`?",
  //     options: [
  //       "A `Serializer` is used for read operations, while a `ModelSerializer` is for write operations.",
  //       "A `ModelSerializer` automatically generates fields and validators from a Django model, reducing boilerplate code.",
  //       "A `ModelSerializer` can only output JSON, while a `Serializer` can output XML.",
  //       "There is no difference; `ModelSerializer` is a deprecated name for `Serializer`.",
  //     ],
  //     correctAnswer:
  //       "A `ModelSerializer` automatically generates fields and validators from a Django model, reducing boilerplate code.",
  //   },
  //   {
  //     id: "fspy36",
  //     question:
  //       "What is an 'idempotent' operation in the context of HTTP methods?",
  //     options: [
  //       "An operation that can be safely executed multiple times without changing the result beyond the initial application (e.g., PUT, DELETE).",
  //       "An operation that is guaranteed to be successful.",
  //       "An operation that does not change the state of the server (e.g., GET, HEAD).",
  //       "An operation that can only be performed once.",
  //     ],
  //     correctAnswer:
  //       "An operation that can be safely executed multiple times without changing the result beyond the initial application (e.g., PUT, DELETE).",
  //   },
  //   {
  //     id: "fspy37",
  //     question:
  //       "In Python, what does the expression `*args` and `**kwargs` in a function signature allow for?",
  //     options: [
  //       "They enforce that all arguments must be named.",
  //       "They are used for pointer arithmetic, similar to C.",
  //       "They allow a function to accept a variable number of positional arguments (`*args`) and keyword arguments (`**kwargs`).",
  //       "They automatically convert all arguments to strings.",
  //     ],
  //     correctAnswer:
  //       "They allow a function to accept a variable number of positional arguments (`*args`) and keyword arguments (`**kwargs`).",
  //   },
  //   {
  //     id: "fspy38",
  //     question:
  //       "In React, when should you use `useLayoutEffect` instead of `useEffect`?",
  //     options: [
  //       "When performing side effects that do not require access to the DOM.",
  //       "For all data fetching operations.",
  //       "When your effect needs to synchronously re-render and measure the DOM before the browser has a chance to paint.",
  //       "They are interchangeable and `useEffect` is always preferred for simplicity.",
  //     ],
  //     correctAnswer:
  //       "When your effect needs to synchronously re-render and measure the DOM before the browser has a chance to paint.",
  //   },
  //   {
  //     id: "fspy39",
  //     question:
  //       "What is the purpose of a reverse proxy like Nginx when deployed in front of a Python WSGI server?",
  //     options: [
  //       "To interpret and execute Python code directly.",
  //       "To handle tasks like serving static files, SSL termination, load balancing, and forwarding requests to the application server.",
  //       "To connect the application directly to the database.",
  //       "To monitor application performance and report errors.",
  //     ],
  //     correctAnswer:
  //       "To handle tasks like serving static files, SSL termination, load balancing, and forwarding requests to the application server.",
  //   },
  //   {
  //     id: "fspy40",
  //     question:
  //       "When using `Docker Compose`, what is the purpose of the `depends_on` option?",
  //     options: [
  //       "It ensures that a service's container is started before another service's container, but it does not wait for the dependency to be 'ready'.",
  //       "It sets up a networking link between the services.",
  //       "It installs the software dependencies of one service into another.",
  //       "It guarantees that the dependent service is fully initialized and ready to accept connections before starting the other service.",
  //     ],
  //     correctAnswer:
  //       "It ensures that a service's container is started before another service's container, but it does not wait for the dependency to be 'ready'.",
  //   },
  //   {
  //     id: "fspy41",
  //     question:
  //       "What does a Python decorator with arguments, like `@my_decorator(arg1, arg2)`, actually return?",
  //     options: [
  //       "The original function, unmodified.",
  //       "A new class instance.",
  //       "A wrapper function that takes the decorated function as its argument.",
  //       "A function that acts as the actual decorator.",
  //     ],
  //     correctAnswer: "A function that acts as the actual decorator.",
  //   },
  //   {
  //     id: "fspy42",
  //     question:
  //       "In CSS Grid, what is the difference between `fr` units and `%` units for defining track sizes?",
  //     options: [
  //       "`fr` units are for rows and `%` units are for columns.",
  //       "`fr` represents a fraction of the available space in the grid container, while `%` relates to the total size of the grid container itself.",
  //       "`%` units cannot be used with the `grid-template-columns` property.",
  //       "They are functionally identical.",
  //     ],
  //     correctAnswer:
  //       "`fr` represents a fraction of the available space in the grid container, while `%` relates to the total size of the grid container itself.",
  //   },
  //   {
  //     id: "fspy43",
  //     question:
  //       "What type of SQL injection vulnerability does using Django's ORM help prevent by default?",
  //     options: [
  //       "Second-order SQL injection.",
  //       "Blind SQL injection.",
  //       "Classic SQL injection, because queries are constructed with parameterized statements.",
  //       "Time-based blind SQL injection.",
  //     ],
  //     correctAnswer:
  //       "Classic SQL injection, because queries are constructed with parameterized statements.",
  //   },
  //   {
  //     id: "fspy44",
  //     question:
  //       "In a CI/CD pipeline, what is the key difference between 'Continuous Integration' and 'Continuous Delivery'?",
  //     options: [
  //       "Integration happens on feature branches, while Delivery happens only on the main branch.",
  //       "Continuous Integration focuses on merging and testing code frequently, while Continuous Delivery ensures that every change that passes tests is automatically prepared for a release to production.",
  //       "They are the same concept.",
  //       "Continuous Integration uses Docker, while Continuous Delivery uses virtual machines.",
  //     ],
  //     correctAnswer:
  //       "Continuous Integration focuses on merging and testing code frequently, while Continuous Delivery ensures that every change that passes tests is automatically prepared for a release to production.",
  //   },
  //   {
  //     id: "fspy45",
  //     question: "What is a 'closure' in JavaScript?",
  //     options: [
  //       "A syntax for creating private methods in classes.",
  //       "A function that has been closed and can no longer be executed.",
  //       "The combination of a function and the lexical environment within which that function was declared, allowing it to access variables from its outer scope even after the outer function has finished executing.",
  //       "An object that contains all global variables.",
  //     ],
  //     correctAnswer:
  //       "The combination of a function and the lexical environment within which that function was declared, allowing it to access variables from its outer scope even after the outer function has finished executing.",
  //   },
  //   {
  //     id: "fspy46",
  //     question: "In Django, how does the `F()` expression work in a queryset?",
  //     options: [
  //       "It allows you to refer to a model field's value directly in the database, enabling atomic updates without pulling the object into Python memory.",
  //       "It is a shortcut for filtering by a ForeignKey's primary key.",
  //       "It fetches the first object that matches the query.",
  //       "It formats a field's value into a specific string representation.",
  //     ],
  //     correctAnswer:
  //       "It allows you to refer to a model field's value directly in the database, enabling atomic updates without pulling the object into Python memory.",
  //   },
  //   {
  //     id: "fspy47",
  //     question:
  //       "What is the primary purpose of server-side rendering (SSR) in a single-page application (SPA) framework like React?",
  //     options: [
  //       "To reduce the load on the server by offloading rendering to the client.",
  //       "To improve initial page load performance and SEO by sending a fully rendered HTML page to the browser.",
  //       "To eliminate the need for JavaScript on the client-side.",
  //       "To make the application work offline.",
  //     ],
  //     correctAnswer:
  //       "To improve initial page load performance and SEO by sending a fully rendered HTML page to the browser.",
  //   },
  //   {
  //     id: "fspy48",
  //     question:
  //       "What is the difference between a `Process` and a `Thread` in Python's `multiprocessing` and `threading` modules?",
  //     options: [
  //       "Threads share the same memory space, while Processes have separate memory spaces.",
  //       "Processes are not affected by the GIL, but Threads are.",
  //       "Threads are suitable for I/O-bound tasks, while Processes are better for CPU-bound tasks in CPython.",
  //       "All of the above.",
  //     ],
  //     correctAnswer: "All of the above.",
  //   },
  //   {
  //     id: "fspy49",
  //     question:
  //       "In a SQL `GROUP BY` clause, what is the purpose of the `HAVING` clause?",
  //     options: [
  //       "It is an alias for the `WHERE` clause and can be used interchangeably.",
  //       "It filters rows before the grouping is applied.",
  //       "It filters the results of a `GROUP BY` clause based on an aggregate function, which cannot be done in the `WHERE` clause.",
  //       "It sorts the grouped results.",
  //     ],
  //     correctAnswer:
  //       "It filters the results of a `GROUP BY` clause based on an aggregate function, which cannot be done in the `WHERE` clause.",
  //   },
  //   {
  //     id: "fspy50",
  //     question: "In Flask, what is the role of a 'Blueprint'?",
  //     options: [
  //       "To define the database schema for the application.",
  //       "A way to organize a group of related views, templates, and static files, making the application more modular and reusable.",
  //       "A tool for automatically generating API documentation.",
  //       "A security feature for defining user roles and permissions.",
  //     ],
  //     correctAnswer:
  //       "A way to organize a group of related views, templates, and static files, making the application more modular and reusable.",
  //   },
  // ],
  // "Data Scientist": [
  //   {
  //     id: "ds1",
  //     question:
  //       "What is the primary motivation for using the kernel trick in Support Vector Machines (SVMs)?",
  //     options: [
  //       "To reduce the number of support vectors and speed up prediction time.",
  //       "To compute the dot products of feature vectors in a higher-dimensional space efficiently, without explicitly transforming the data, allowing for non-linear decision boundaries.",
  //       "To enforce a hard margin for better classification on linearly separable data.",
  //       "To apply L2 regularization to the model's weight vector.",
  //     ],
  //     correctAnswer:
  //       "To compute the dot products of feature vectors in a higher-dimensional space efficiently, without explicitly transforming the data, allowing for non-linear decision boundaries.",
  //   },
  //   {
  //     id: "apt1",
  //     question:
  //       "A trader mixes 26 kg of rice at Rs. 20 per kg with 30 kg of rice of another variety at Rs. 36 per kg and sells the mixture at Rs. 30 per kg. What is his profit percent?",
  //     options: ["No profit, no loss", "5%", "8%", "10%"],
  //     correctAnswer: "5%",
  //   },
  //   {
  //     id: "ds2",
  //     question:
  //       "In the context of Gradient Boosting Machines (GBMs), how does the 'learning rate' (or shrinkage) parameter help in preventing overfitting?",
  //     options: [
  //       "It increases the complexity of each individual decision tree in the ensemble.",
  //       "It randomly drops a fraction of features at each split, similar to Random Forest.",
  //       "It reduces the contribution of each tree to the final model, forcing the algorithm to use more trees to explain the variance, which often leads to better generalization.",
  //       "It applies post-pruning to each tree after it has been fully grown.",
  //     ],
  //     correctAnswer:
  //       "It reduces the contribution of each tree to the final model, forcing the algorithm to use more trees to explain the variance, which often leads to better generalization.",
  //   },
  //   {
  //     id: "apt2",
  //     question: "Find the next term in the series: 5, 6, 14, 45, 184, ?",
  //     options: ["925", "845", "905", "945"],
  //     correctAnswer: "925",
  //   },
  //   {
  //     id: "ds3",
  //     question:
  //       "What is the key difference between L1 (Lasso) and L2 (Ridge) regularization in a linear model?",
  //     options: [
  //       "L1 can shrink some coefficients to exactly zero, performing feature selection, while L2 only shrinks them towards zero.",
  //       "L2 is computationally less expensive than L1 and is therefore preferred for large datasets.",
  //       "L1 is used for classification problems, while L2 is used for regression problems.",
  //       "L2 regularization is more effective at handling multicollinearity than L1.",
  //     ],
  //     correctAnswer:
  //       "L1 can shrink some coefficients to exactly zero, performing feature selection, while L2 only shrinks them towards zero.",
  //   },
  //   {
  //     id: "ds4",
  //     question:
  //       "You perform a hypothesis test and get a p-value of 0.03. With a significance level (alpha) of 0.05, you reject the null hypothesis. What does this p-value signify?",
  //     options: [
  //       "There is a 3% chance that the null hypothesis is true.",
  //       "There is a 3% chance of observing the data you have, assuming the alternative hypothesis is true.",
  //       "There is a 97% chance that the alternative hypothesis is true.",
  //       "Assuming the null hypothesis is true, there is a 3% probability of observing a result at least as extreme as the one in your data.",
  //     ],
  //     correctAnswer:
  //       "Assuming the null hypothesis is true, there is a 3% probability of observing a result at least as extreme as the one in your data.",
  //   },
  //   {
  //     id: "apt3",
  //     question:
  //       "A, B and C enter into a partnership. A invests 3 times as much as B invests and B invests two-third of what C invests. At the end of the year, the profit earned is Rs. 6600. What is the share of B?",
  //     options: ["Rs. 1200", "Rs. 1800", "Rs. 2400", "Rs. 3600"],
  //     correctAnswer: "Rs. 1200",
  //   },
  //   {
  //     id: "ds5",
  //     question:
  //       "What is the primary advantage of using the AUC (Area Under the ROC Curve) metric over accuracy for an imbalanced classification problem?",
  //     options: [
  //       "AUC is easier to calculate and interpret than accuracy.",
  //       "AUC is scale-invariant and classification-threshold-invariant, providing a better measure of a model's ability to distinguish between classes regardless of the chosen threshold.",
  //       "AUC works for multi-class classification, whereas accuracy is only for binary problems.",
  //       "AUC penalizes false positives more heavily than accuracy.",
  //     ],
  //     correctAnswer:
  //       "AUC is scale-invariant and classification-threshold-invariant, providing a better measure of a model's ability to distinguish between classes regardless of the chosen threshold.",
  //   },
  //   {
  //     id: "ds6",
  //     question:
  //       "In a deep neural network, what is the 'vanishing gradient' problem?",
  //     options: [
  //       "A problem where the model's weights converge to zero, resulting in no learning.",
  //       "A situation where the gradients of the loss function approach infinity, causing unstable training.",
  //       "The issue where gradients become extremely small as they are backpropagated through many layers, causing the weights of the initial layers to update very slowly or not at all.",
  //       "A regularization technique that intentionally sets some gradients to zero to prevent overfitting.",
  //     ],
  //     correctAnswer:
  //       "The issue where gradients become extremely small as they are backpropagated through many layers, causing the weights of the initial layers to update very slowly or not at all.",
  //   },
  //   {
  //     id: "apt4",
  //     question:
  //       "In a group of 6 boys and 4 girls, four children are to be selected. In how many different ways can they be selected such that at least one boy should be there?",
  //     options: ["159", "209", "201", "212"],
  //     correctAnswer: "209",
  //   },
  //   {
  //     id: "ds7",
  //     question:
  //       "What is a key difference between Principal Component Analysis (PCA) and t-SNE (t-Distributed Stochastic Neighbor Embedding)?",
  //     options: [
  //       "PCA is a supervised technique, while t-SNE is unsupervised.",
  //       "PCA is primarily a dimensionality reduction technique focused on preserving global variance, while t-SNE is a visualization technique focused on preserving local similarities in high-dimensional data.",
  //       "t-SNE is deterministic, while PCA involves a random component.",
  //       "PCA can only be used for numerical data, while t-SNE also works for categorical data.",
  //     ],
  //     correctAnswer:
  //       "PCA is primarily a dimensionality reduction technique focused on preserving global variance, while t-SNE is a visualization technique focused on preserving local similarities in high-dimensional data.",
  //   },
  //   {
  //     id: "ds8",
  //     question: "What is the bias-variance tradeoff in machine learning?",
  //     options: [
  //       "The tradeoff between the training time of a model and its prediction accuracy.",
  //       "The tradeoff where a model with low bias (high complexity) tends to have high variance (overfitting), and a model with high bias (low complexity) tends to have low variance (underfitting).",
  //       "The tradeoff between using a biased dataset and achieving a model with low variance.",
  //       "The tradeoff between precision and recall in a classification model.",
  //     ],
  //     correctAnswer:
  //       "The tradeoff where a model with low bias (high complexity) tends to have high variance (overfitting), and a model with high bias (low complexity) tends to have low variance (underfitting).",
  //   },
  //   {
  //     id: "apt5",
  //     question:
  //       "A man can row at 5 kmph in still water. If the velocity of the current is 1 kmph and it takes him 1 hour to row to a place and come back, how far is the place?",
  //     options: ["2.4 km", "2.5 km", "3 km", "3.2 km"],
  //     correctAnswer: "2.4 km",
  //   },
  //   {
  //     id: "ds9",
  //     question:
  //       "How does a Random Forest model reduce variance compared to a single Decision Tree?",
  //     options: [
  //       "By using deeper, more complex trees.",
  //       "By applying a strong pruning algorithm to each tree.",
  //       "By building multiple trees on bootstrapped samples of the data and averaging their predictions, which reduces the overall model's sensitivity to the specific training set.",
  //       "By using a different impurity measure, such as Gini impurity instead of entropy.",
  //     ],
  //     correctAnswer:
  //       "By building multiple trees on bootstrapped samples of the data and averaging their predictions, which reduces the overall model's sensitivity to the specific training set.",
  //   },
  //   {
  //     id: "ds10",
  //     question: "What is Simpson's Paradox?",
  //     options: [
  //       "The paradox that a model can have high accuracy but be practically useless due to class imbalance.",
  //       "A statistical phenomenon where a trend appears in several different groups of data but disappears or reverses when these groups are combined.",
  //       "The paradox that adding more features to a model can sometimes decrease its performance.",
  //       "The observation that the mean of a dataset can be heavily skewed by outliers.",
  //     ],
  //     correctAnswer:
  //       "A statistical phenomenon where a trend appears in several different groups of data but disappears or reverses when these groups are combined.",
  //   },
  //   {
  //     id: "apt6",
  //     question:
  //       "Statements: Some dogs are cats. All cats are pigs. Conclusions: 1. Some dogs are pigs. 2. Some pigs are cats.",
  //     options: [
  //       "Only conclusion 1 follows",
  //       "Only conclusion 2 follows",
  //       "Both 1 and 2 follow",
  //       "Neither 1 nor 2 follows",
  //     ],
  //     correctAnswer: "Both 1 and 2 follow",
  //   },
  //   {
  //     id: "ds11",
  //     question:
  //       "What is the main purpose of using cross-validation when evaluating a machine learning model?",
  //     options: [
  //       "To speed up the training process by using smaller subsets of data.",
  //       "To obtain a more robust estimate of the model's performance on unseen data by training and testing on different partitions of the dataset.",
  //       "To automatically tune the hyperparameters of the model.",
  //       "To reduce the bias of the model by ensuring all data points are used for training.",
  //     ],
  //     correctAnswer:
  //       "To obtain a more robust estimate of the model's performance on unseen data by training and testing on different partitions of the dataset.",
  //   },
  //   {
  //     id: "ds12",
  //     question:
  //       "Which of the following is an assumption of linear regression that, if violated, can be detected by plotting residuals against predicted values?",
  //     options: [
  //       "Normality of residuals",
  //       "Independence of errors",
  //       "Homoscedasticity (constant variance of errors)",
  //       "No or little multicollinearity",
  //     ],
  //     correctAnswer: "Homoscedasticity (constant variance of errors)",
  //   },
  //   {
  //     id: "apt7",
  //     question:
  //       "A clock shows 8 o'clock in the morning. Through how many degrees will the hour hand rotate when the clock shows 2 o'clock in the afternoon?",
  //     options: ["144°", "150°", "168°", "180°"],
  //     correctAnswer: "180°",
  //   },
  //   {
  //     id: "ds13",
  //     question: "In the context of A/B testing, what is a Type I error?",
  //     options: [
  //       "Failing to reject the null hypothesis when it is false (a false negative).",
  //       "Incorrectly rejecting the null hypothesis when it is true (a false positive).",
  //       "Concluding that the variance of the two groups is equal when it is not.",
  //       "Stopping the test too early, leading to a biased result.",
  //     ],
  //     correctAnswer:
  //       "Incorrectly rejecting the null hypothesis when it is true (a false positive).",
  //   },
  //   {
  //     id: "ds14",
  //     question:
  //       "What is the primary difference between the K-Means and DBSCAN clustering algorithms?",
  //     options: [
  //       "K-Means is a hierarchical algorithm, while DBSCAN is a partitional algorithm.",
  //       "K-Means requires the number of clusters (k) to be specified beforehand and assumes clusters are spherical, while DBSCAN can find arbitrarily shaped clusters and identify noise points.",
  //       "DBSCAN is significantly faster than K-Means on large datasets.",
  //       "K-Means can only be used with numerical data, whereas DBSCAN can handle categorical data.",
  //     ],
  //     correctAnswer:
  //       "K-Means requires the number of clusters (k) to be specified beforehand and assumes clusters are spherical, while DBSCAN can find arbitrarily shaped clusters and identify noise points.",
  //   },
  //   {
  //     id: "apt8",
  //     question:
  //       "If P means 'division', T means 'addition', M means 'subtraction' and D means 'multiplication', then what will be the value of the expression: 12 M 12 D 28 P 7 T 15?",
  //     options: ["-30", "-15", "15", "-21"],
  //     correctAnswer: "-21",
  //   },
  //   {
  //     id: "ds15",
  //     question:
  //       "What does it mean if a feature has high 'feature importance' in a tree-based model like a Random Forest?",
  //     options: [
  //       "The feature has a high correlation with the target variable.",
  //       "The feature was used for splits most frequently and/or resulted in the largest reduction in impurity (e.g., Gini impurity) across all trees in the forest.",
  //       "The feature is guaranteed to have a causal relationship with the target.",
  //       "The feature has a very low number of missing values.",
  //     ],
  //     correctAnswer:
  //       "The feature was used for splits most frequently and/or resulted in the largest reduction in impurity (e.g., Gini impurity) across all trees in the forest.",
  //   },
  //   {
  //     id: "ds16",
  //     question:
  //       "In a Convolutional Neural Network (CNN), what is the main purpose of a pooling layer (e.g., Max Pooling)?",
  //     options: [
  //       "To introduce non-linearity into the model.",
  //       "To reduce the spatial dimensions (width and height) of the feature maps, which helps to reduce computational complexity and control overfitting.",
  //       "To increase the number of feature maps (channels).",
  //       "To perform feature extraction by applying convolutional filters.",
  //     ],
  //     correctAnswer:
  //       "To reduce the spatial dimensions (width and height) of the feature maps, which helps to reduce computational complexity and control overfitting.",
  //   },
  //   {
  //     id: "apt9",
  //     question:
  //       "The present ages of three persons are in proportions 4 : 7 : 9. Eight years ago, the sum of their ages was 56. Find their present ages.",
  //     options: ["16, 28, 36", "20, 35, 45", "8, 20, 28", "20, 28, 36"],
  //     correctAnswer: "16, 28, 36",
  //   },
  //   {
  //     id: "ds17",
  //     question:
  //       "Why is it often necessary to scale features before training a model like an SVM or using a technique like PCA?",
  //     options: [
  //       "To convert all features into a normal distribution.",
  //       "To reduce the number of features in the dataset.",
  //       "Because these algorithms are sensitive to the scale of the input features; features with larger ranges can dominate the objective function or principal components, leading to biased results.",
  //       "To handle categorical features that have been one-hot encoded.",
  //     ],
  //     correctAnswer:
  //       "Because these algorithms are sensitive to the scale of the input features; features with larger ranges can dominate the objective function or principal components, leading to biased results.",
  //   },
  //   {
  //     id: "ds18",
  //     question:
  //       "What is the difference between online learning and batch learning?",
  //     options: [
  //       "Online learning is for web-based applications, while batch learning is for offline data processing.",
  //       "Online learning updates the model incrementally as new data points arrive, while batch learning trains the model on the entire dataset at once.",
  //       "Batch learning is always faster than online learning.",
  //       "Online learning can only be used for regression, not classification.",
  //     ],
  //     correctAnswer:
  //       "Online learning updates the model incrementally as new data points arrive, while batch learning trains the model on the entire dataset at once.",
  //   },
  //   {
  //     id: "apt10",
  //     question:
  //       "A train passes a station platform in 36 seconds and a man standing on the platform in 20 seconds. If the speed of the train is 54 km/hr, what is the length of the platform?",
  //     options: ["220 m", "240 m", "260 m", "280 m"],
  //     correctAnswer: "240 m",
  //   },
  //   {
  //     id: "ds19",
  //     question:
  //       "When building a recommender system, what is the 'cold start' problem?",
  //     options: [
  //       "The difficulty in training the model when the dataset is very small.",
  //       "The problem of making recommendations for new users or new items for which the system has no historical interaction data.",
  //       "The technical challenge of deploying a large recommendation model on a server for the first time.",
  //       "The tendency for recommender systems to recommend only popular items.",
  //     ],
  //     correctAnswer:
  //       "The problem of making recommendations for new users or new items for which the system has no historical interaction data.",
  //   },
  //   {
  //     id: "ds20",
  //     question:
  //       "What is the purpose of the 'dropout' technique in training neural networks?",
  //     options: [
  //       "To speed up training by dropping entire layers from the network.",
  //       "A regularization technique where randomly selected neurons are ignored during training for each batch, which helps prevent co-adaptation of neurons and reduces overfitting.",
  //       "To remove a fixed percentage of the training data that are considered outliers.",
  //       "To reduce the learning rate adaptively during training.",
  //     ],
  //     correctAnswer:
  //       "A regularization technique where randomly selected neurons are ignored during training for each batch, which helps prevent co-adaptation of neurons and reduces overfitting.",
  //   },
  //   {
  //     id: "apt11",
  //     question:
  //       "A box contains 8 red, 7 blue and 6 green balls. One ball is picked up randomly. What is the probability that it is neither red nor green?",
  //     options: ["1/3", "3/4", "7/19", "8/21"],
  //     correctAnswer: "1/3",
  //   },
  //   {
  //     id: "ds21",
  //     question:
  //       "What is the key assumption of the Naive Bayes classifier that makes it 'naive'?",
  //     options: [
  //       "It assumes that the data is normally distributed.",
  //       "It assumes that all features are independent of each other given the class, which is often not true in reality.",
  //       "It assumes that all features are discrete, not continuous.",
  //       "It assumes that the number of data points for each class is perfectly balanced.",
  //     ],
  //     correctAnswer:
  //       "It assumes that all features are independent of each other given the class, which is often not true in reality.",
  //   },
  //   {
  //     id: "ds22",
  //     question:
  //       "In time series analysis, what is 'stationarity' and why is it important?",
  //     options: [
  //       "A property where the time series has a clear, predictable trend. It's important for long-term forecasting.",
  //       "A property where the statistical properties of the series (like mean and variance) are constant over time. It's an underlying assumption for many forecasting models like ARIMA.",
  //       "A method for decomposing a time series into its trend, seasonal, and residual components.",
  //       "A measure of the correlation between a time series and a lagged version of itself.",
  //     ],
  //     correctAnswer:
  //       "A property where the statistical properties of the series (like mean and variance) are constant over time. It's an underlying assumption for many forecasting models like ARIMA.",
  //   },
  //   {
  //     id: "ds23",
  //     question:
  //       "You have a multi-class classification problem with 5 classes. What is the baseline accuracy you should aim to beat?",
  //     options: [
  //       "50%",
  //       "The accuracy achieved by a logistic regression model.",
  //       "The accuracy of predicting the majority class for every instance.",
  //       "95%",
  //     ],
  //     correctAnswer:
  //       "The accuracy of predicting the majority class for every instance.",
  //   },
  //   {
  //     id: "apt12",
  //     question:
  //       "Two numbers are respectively 20% and 50% more than a third number. The ratio of the two numbers is:",
  //     options: ["2:5", "3:5", "4:5", "6:7"],
  //     correctAnswer: "4:5",
  //   },
  //   {
  //     id: "ds24",
  //     question:
  //       "What is the difference between a bagging and a boosting ensemble method?",
  //     options: [
  //       "Bagging is used for classification, boosting for regression.",
  //       "Bagging trains models in parallel on different subsets of data, while boosting trains models sequentially, with each new model attempting to correct the errors of the previous one.",
  //       "Boosting is less prone to overfitting than bagging.",
  //       "Bagging can only use decision trees, while boosting can use any type of model.",
  //     ],
  //     correctAnswer:
  //       "Bagging trains models in parallel on different subsets of data, while boosting trains models sequentially, with each new model attempting to correct the errors of the previous one.",
  //   },
  //   {
  //     id: "ds25",
  //     question: "What statistical concept is Bayes' Theorem based on?",
  //     options: [
  //       "Frequentist probability",
  //       "The law of large numbers",
  //       "Conditional probability",
  //       "The central limit theorem",
  //     ],
  //     correctAnswer: "Conditional probability",
  //   },
  //   {
  //     id: "ds26",
  //     question:
  //       "When would you prefer using a Mean Absolute Error (MAE) loss function over a Mean Squared Error (MSE) loss function in a regression problem?",
  //     options: [
  //       "When the target variable is normally distributed.",
  //       "When the dataset is very large.",
  //       "When the dataset contains significant outliers, as MAE is less sensitive to them than MSE.",
  //       "When you want to penalize large errors more heavily.",
  //     ],
  //     correctAnswer:
  //       "When the dataset contains significant outliers, as MAE is less sensitive to them than MSE.",
  //   },
  //   {
  //     id: "apt13",
  //     question:
  //       "A can finish a work in 18 days and B can do the same work in 15 days. B worked for 10 days and left the job. In how many days, A alone can finish the remaining work?",
  //     options: ["5", "5.5", "6", "8"],
  //     correctAnswer: "6",
  //   },
  //   {
  //     id: "ds27",
  //     question:
  //       "What is the primary risk of data leakage in a machine learning pipeline?",
  //     options: [
  //       "The model taking too long to train on a large dataset.",
  //       "The model becoming too complex and difficult to interpret.",
  //       "When information from outside the training data is used to create the model, leading to an overly optimistic evaluation of its performance (e.g., scaling data before splitting into train/test sets).",
  //       "The model's predictions violating data privacy regulations.",
  //     ],
  //     correctAnswer:
  //       "When information from outside the training data is used to create the model, leading to an overly optimistic evaluation of its performance (e.g., scaling data before splitting into train/test sets).",
  //   },
  //   {
  //     id: "ds28",
  //     question:
  //       "What does a Gini impurity of 0 in a node of a decision tree signify?",
  //     options: [
  //       "The node contains an equal mix of all classes.",
  //       "The node is a leaf node, and all data points within it belong to a single class (it is perfectly pure).",
  //       "The node is the root of the tree.",
  //       "The split at this node did not improve the model's performance.",
  //     ],
  //     correctAnswer:
  //       "The node is a leaf node, and all data points within it belong to a single class (it is perfectly pure).",
  //   },
  //   {
  //     id: "ds29",
  //     question:
  //       "What is a major advantage of a multi-armed bandit approach over a traditional A/B test for optimizing a website?",
  //     options: [
  //       "It is easier to implement and requires no statistical knowledge.",
  //       "It dynamically allocates more traffic to the better-performing variation during the test, minimizing regret (lost conversions) compared to a fixed allocation in A/B testing.",
  //       "It can test more variations simultaneously than an A/B test.",
  //       "It guarantees finding a statistically significant result faster.",
  //     ],
  //     correctAnswer:
  //       "It dynamically allocates more traffic to the better-performing variation during the test, minimizing regret (lost conversions) compared to a fixed allocation in A/B testing.",
  //   },
  //   {
  //     id: "apt14",
  //     question:
  //       "How many kilograms of sugar costing Rs. 9 per kg must be mixed with 27 kg of sugar costing Rs. 7 per kg so that there may be a gain of 10% by selling the mixture at Rs. 9.24 per kg?",
  //     options: ["36 kg", "42 kg", "54 kg", "63 kg"],
  //     correctAnswer: "63 kg",
  //   },
  //   {
  //     id: "ds30",
  //     question:
  //       "In Natural Language Processing (NLP), what is the primary difference between TF-IDF and word embeddings like Word2Vec?",
  //     options: [
  //       "TF-IDF is a simple frequency-based vectorization method that doesn't capture semantic meaning, while Word2Vec learns dense vector representations that capture semantic relationships between words.",
  //       "Word2Vec can only be used for English, while TF-IDF is language-agnostic.",
  //       "TF-IDF is a deep learning model, whereas Word2Vec is a statistical method.",
  //       "TF-IDF produces very short vectors, while Word2Vec produces very long, sparse vectors.",
  //     ],
  //     correctAnswer:
  //       "TF-IDF is a simple frequency-based vectorization method that doesn't capture semantic meaning, while Word2Vec learns dense vector representations that capture semantic relationships between words.",
  //   },
  //   {
  //     id: "ds31",
  //     question:
  //       "What is heteroscedasticity, and why is it a problem in linear regression?",
  //     options: [
  //       "It is when the independent variables are highly correlated; it makes coefficient estimates unstable.",
  //       "It refers to the situation where the variance of the residuals is not constant across all levels of the independent variables, violating a key assumption and potentially making standard errors of the coefficients unreliable.",
  //       "It is when the relationship between the independent and dependent variables is non-linear, leading to a poor model fit.",
  //       "It is the presence of significant outliers in the data.",
  //     ],
  //     correctAnswer:
  //       "It refers to the situation where the variance of the residuals is not constant across all levels of the independent variables, violating a key assumption and potentially making standard errors of the coefficients unreliable.",
  //   },
  //   {
  //     id: "apt15",
  //     question:
  //       "Pointing to a photograph, a man said, 'I have no brother or sister but that man's father is my father's son.' Whose photograph was it?",
  //     options: ["His own", "His Son's", "His Father's", "His Nephew's"],
  //     correctAnswer: "His Son's",
  //   },
  //   {
  //     id: "ds32",
  //     question:
  //       "What is the purpose of an activation function (like ReLU or Sigmoid) in a neural network?",
  //     options: [
  //       "To normalize the inputs to a neuron.",
  //       "To introduce non-linearity into the network, allowing it to learn complex patterns that a simple linear model cannot.",
  //       "To calculate the loss of the network's predictions.",
  //       "To regularize the network and prevent overfitting.",
  //     ],
  //     correctAnswer:
  //       "To introduce non-linearity into the network, allowing it to learn complex patterns that a simple linear model cannot.",
  //   },
  //   {
  //     id: "ds33",
  //     question: "What is a 'stratified sample' and when would you use it?",
  //     options: [
  //       "A sample where every member of the population has an equal chance of being selected.",
  //       "A sample created by dividing the population into subgroups (strata) based on a shared characteristic and then taking a random sample from each subgroup, ensuring representation of all groups.",
  //       "A sample selected at regular intervals from an ordered list of the population.",
  //       "A sample created by selecting entire clusters or groups from the population.",
  //     ],
  //     correctAnswer:
  //       "A sample created by dividing the population into subgroups (strata) based on a shared characteristic and then taking a random sample from each subgroup, ensuring representation of all groups.",
  //   },
  //   {
  //     id: "ds34",
  //     question:
  //       "When comparing two classification models, Model A has higher Precision and Model B has higher Recall. In which scenario would you prefer Model B?",
  //     options: [
  //       "Spam email detection, where you want to minimize the number of legitimate emails marked as spam.",
  //       "Medical diagnosis for a serious disease, where it is critical to identify as many actual cases as possible, even at the cost of some false alarms.",
  //       "A system that recommends products to users, where relevance is key.",
  //       "A system that must be highly accurate and has no preference between false positives and false negatives.",
  //     ],
  //     correctAnswer:
  //       "Medical diagnosis for a serious disease, where it is critical to identify as many actual cases as possible, even at the cost of some false alarms.",
  //   },
  //   {
  //     id: "ds35",
  //     question: "What does a lift chart in a classification model measure?",
  //     options: [
  //       "The overall accuracy of the model.",
  //       "The model's performance in ranking predictions from most to least likely, showing how much more likely we are to find positive cases in a decile compared to random chance.",
  //       "The tradeoff between the true positive rate and the false positive rate.",
  //       "The computational resources required to train the model.",
  //     ],
  //     correctAnswer:
  //       "The model's performance in ranking predictions from most to least likely, showing how much more likely we are to find positive cases in a decile compared to random chance.",
  //   },
  // ],

  "Talent Recruitment Specialist": [
  
  {
    id: "trs1",
    question: "What is the primary role of a Talent Recruitment Specialist?",
    options: [
      "Payroll management",
      "Employee engagement",
      "Sourcing and hiring candidates",
      "Training employees",
    ],
    correctAnswer: "Sourcing and hiring candidates",
  },
  {
    id: "trs2",
    question: "A candidate has multiple offers. What should a recruiter do?",
    options: [
      "Pressure the candidate",
      "Withdraw offer",
      "Maintain transparent communication",
      "Ignore follow-ups",
    ],
    correctAnswer: "Maintain transparent communication",
  },
  {
    id: "trs3",
    question: "What is the BEST way to reduce offer dropouts?",
    options: [
      "Delay offer release",
      "Strong pre-offer engagement",
      "Lower salary",
      "Shorter interview",
    ],
    correctAnswer: "Strong pre-offer engagement",
  },
  {
    id: "trs4",
    question: "If a candidate clears interviews but exceeds budget, what should you do?",
    options: [
      "Reject immediately",
      "Negotiate with candidate and hiring manager",
      "Put on hold forever",
      "Change job role",
    ],
    correctAnswer: "Negotiate with candidate and hiring manager",
  },
  {
    id: "trs5",
    question: "What indicates a strong hiring pipeline?",
    options: [
      "Many interviews scheduled",
      "Candidates ready for future roles",
      "Fast offer release",
      "High salary offers",
    ],
    correctAnswer: "Candidates ready for future roles",
  },
  {
    id: "trs6",
    question: "Find the odd one out: Recruiter, Manager, Developer, Chair",
    options: ["Recruiter", "Manager", "Developer", "Chair"],
    correctAnswer: "Chair",
  },
  {
    id: "trs7",
    question: "Complete the series: 2, 6, 12, 20, 30, ?",
    options: ["40", "42", "44", "45"],
    correctAnswer: "42",
  },
  {
    id: "trs8",
    question: "Which word does NOT belong: Resume, Offer, Join, Train",
    options: ["Resume", "Offer", "Join", "Train"],
    correctAnswer: "Train",
  },
  {
    id: "trs9",
    question: "If all developers are employees and some employees are managers, which is TRUE?",
    options: [
      "All developers are managers",
      "Some developers may be managers",
      "No developers are employees",
      "All managers are developers",
    ],
    correctAnswer: "Some developers may be managers",
  },
  {
    id: "trs10",
    question: "Choose the word most similar to 'ATTRITION':",
    options: ["Turnover", "Hiring", "Promotion", "Onboarding"],
    correctAnswer: "Turnover",
  },

  {
    id: "trs11",
    question: "What does HTML stand for?",
    options: [
      "Hyper Trainer Markup Language",
      "Hyper Text Markup Language",
      "High Text Machine Language",
      "Hyper Tool Markup Language",
    ],
    correctAnswer: "Hyper Text Markup Language",
  },
  {
    id: "trs12",
    question: "Which is a frontend framework/library?",
    options: ["Java", "React", "MongoDB", "Node.js"],
    correctAnswer: "React",
  },
  {
    id: "trs13",
    question: "What is CSS used for?",
    options: [
      "Database management",
      "Styling web pages",
      "Server-side logic",
      "API development",
    ],
    correctAnswer: "Styling web pages",
  },
  {
    id: "trs14",
    question: "Which language runs in the browser?",
    options: ["Python", "Java", "JavaScript", "C++"],
    correctAnswer: "JavaScript",
  },
  {
    id: "trs15",
    question: "What is responsive design?",
    options: [
      "Fast website loading",
      "Design adapting to different devices",
      "Backend optimization",
      "Database scaling",
    ],
    correctAnswer: "Design adapting to different devices",
  },
  {
    id: "trs16",
    question: "Which CSS framework is utility-first?",
    options: ["Bootstrap", "Tailwind CSS", "Material UI", "Ant Design"],
    correctAnswer: "Tailwind CSS",
  },
  {
    id: "trs17",
    question: "What is JSX?",
    options: [
      "Java XML",
      "JavaScript syntax extension",
      "JSON format",
      "Backend template",
    ],
    correctAnswer: "JavaScript syntax extension",
  },
  {
    id: "trs18",
    question: "Which tool is commonly used for frontend bundling?",
    options: ["Webpack", "Docker", "Jenkins", "MongoDB"],
    correctAnswer: "Webpack",
  },
  {
    id: "trs19",
    question: "What does SPA stand for?",
    options: [
      "Single Page Application",
      "Server Programming App",
      "Software Process Architecture",
      "System Page Access",
    ],
    correctAnswer: "Single Page Application",
  },
  {
    id: "trs20",
    question: "Which HTML tag is used for links?",
    options: ["<link>", "<a>", "<href>", "<nav>"],
    correctAnswer: "<a>",
  },
  {
    id: "trs21",
    question: "Which is NOT a frontend technology?",
    options: ["HTML", "CSS", "JavaScript", "MongoDB"],
    correctAnswer: "MongoDB",
  },
  {
    id: "trs22",
    question: "What is the virtual DOM associated with?",
    options: ["Angular", "React", "Vue", "jQuery"],
    correctAnswer: "React",
  },
  {
    id: "trs23",
    question: "Which unit is used in CSS for responsive sizing?",
    options: ["px", "em", "rem", "All of the above"],
    correctAnswer: "All of the above",
  },
  {
    id: "trs24",
    question: "Which file typically contains frontend dependencies?",
    options: ["index.html", "package.json", "README.md", "server.js"],
    correctAnswer: "package.json",
  },
  {
    id: "trs25",
    question: "What does SEO stand for?",
    options: [
      "System Engine Output",
      "Search Engine Optimization",
      "Search Enhancement Operation",
      "Site Execution Order",
    ],
    correctAnswer: "Search Engine Optimization",
  },
  {
    id: "trs26",
    question: "Which tool is used for UI design?",
    options: ["Figma", "Git", "Docker", "AWS"],
    correctAnswer: "Figma",
  },
  {
    id: "trs27",
    question: "What does NPM stand for?",
    options: [
      "Node Package Manager",
      "New Programming Model",
      "Network Package Module",
      "Node Process Manager",
    ],
    correctAnswer: "Node Package Manager",
  },
  {
    id: "trs28",
    question: "Which frontend framework is developed by Google?",
    options: ["React", "Vue", "Angular", "Svelte"],
    correctAnswer: "Angular",
  },
  {
    id: "trs29",
    question: "What is accessibility (a11y)?",
    options: [
      "Making apps faster",
      "Making apps usable for everyone",
      "Improving SEO",
      "Reducing bundle size",
    ],
    correctAnswer: "Making apps usable for everyone",
  },
  {
    id: "trs30",
    question: "Which tag is used to display images?",
    options: ["<image>", "<img>", "<pic>", "<src>"],
    correctAnswer: "<img>",
  },


  {
    id: "trs31",
    question: "Which is a backend runtime?",
    options: ["React", "Node.js", "CSS", "HTML"],
    correctAnswer: "Node.js",
  },
  {
    id: "trs32",
    question: "What does SQL stand for?",
    options: [
      "Structured Query Language",
      "Simple Query Language",
      "Standard Query Layer",
      "Sequential Query Logic",
    ],
    correctAnswer: "Structured Query Language",
  },
  {
    id: "trs33",
    question: "Which database is NoSQL?",
    options: ["MySQL", "PostgreSQL", "MongoDB", "Oracle"],
    correctAnswer: "MongoDB",
  },
  {
    id: "trs34",
    question: "What is REST API?",
    options: [
      "Remote Execution Tool",
      "Representational State Transfer",
      "Resource Storage Technique",
      "Real-time Service Tool",
    ],
    correctAnswer: "Representational State Transfer",
  },
  {
    id: "trs35",
    question: "Which HTTP method is used to fetch data?",
    options: ["POST", "PUT", "GET", "DELETE"],
    correctAnswer: "GET",
  },
  {
    id: "trs36",
    question: "What is authentication?",
    options: [
      "Verifying user identity",
      "Storing data",
      "Authorizing roles",
      "Encrypting files",
    ],
    correctAnswer: "Verifying user identity",
  },
  {
    id: "trs37",
    question: "Which tool is used for version control?",
    options: ["Jira", "Git", "Docker", "AWS"],
    correctAnswer: "Git",
  },
  {
    id: "trs38",
    question: "What is Docker used for?",
    options: [
      "UI development",
      "Containerizing applications",
      "Writing APIs",
      "Database queries",
    ],
    correctAnswer: "Containerizing applications",
  },
  {
    id: "trs39",
    question: "What is a production environment?",
    options: [
      "Testing setup",
      "Live system for users",
      "Local development",
      "Training platform",
    ],
    correctAnswer: "Live system for users",
  },
  {
    id: "trs40",
    question: "What does CI/CD mean?",
    options: [
      "Code Integration / Code Deployment",
      "Continuous Integration / Continuous Deployment",
      "Central Integration / Central Deployment",
      "Continuous Improvement / Coding",
    ],
    correctAnswer: "Continuous Integration / Continuous Deployment",
  },
  {
    id: "trs41",
    question: "If all recruiters are HRs and all HRs are employees, which statement is TRUE?",
    options: [
      "All employees are recruiters",
      "Some employees are recruiters",
      "All recruiters are employees",
      "No recruiters are employees",
    ],
    correctAnswer: "All recruiters are employees",
  },
  {
    id: "trs42",
    question: "Choose the word that is most similar to 'HIRE':",
    options: ["Recruit", "Fire", "Train", "Evaluate"],
    correctAnswer: "Recruit",
  },
  {
    id: "trs43",
    question: "Which option does NOT belong: Resume, Offer letter, Interview, Job Description?",
    options: ["Resume", "Offer letter", "Interview", "Job Description"],
    correctAnswer: "Interview",
  },
  {
    id: "trs44",
    question: "Find the odd one out: Permanent, Contract, Internship, Salary",
    options: ["Permanent", "Contract", "Internship", "Salary"],
    correctAnswer: "Salary",
  },
  {
    id: "trs45",
    question: "Complete the analogy: Interview is to Candidate as Meeting is to ____?",
    options: ["Manager", "Discussion", "Agenda", "Employee"],
    correctAnswer: "Employee",
  },
  {
    id: "trs46",
    question: "If every recruiter completes 5 interviews per day, how many interviews do 3 recruiters complete in 2 days?",
    options: ["15", "20", "25", "30"],
    correctAnswer: "30",
  },
  {
    id: "trs47",
    question: "A recruiter must hire either a designer or a developer. If a developer is hired, the designer cannot be. This is an example of:",
    options: [
      "Conditional statement",
      "Contradiction",
      "Parallel hiring",
      "Independent hiring",
    ],
    correctAnswer: "Conditional statement",
  },
  {
    id: "trs48",
    question: "Find the missing letter: A, C, F, J, O, ?",
    options: ["T", "U", "S", "P"],
    correctAnswer: "U",
  },
  {
    id: "trs49",
    question: "Select the word that is opposite in meaning to 'Flexible':",
    options: ["Rigid", "Adaptive", "Bendable", "Supple"],
    correctAnswer: "Rigid",
  },
  {
    id: "trs50",
    question: "If the first two statements are true, is the final statement true?\n1) All interns are candidates.\n2) Some candidates are recruiters.\nStatement: Some interns are recruiters.",
    options: ["True", "False", "Cannot be determined", "None of the above"],
    correctAnswer: "Cannot be determined",
  },
],

};

