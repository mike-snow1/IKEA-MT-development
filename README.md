# IKEA-MT

This is the repository for developing IKEA-MT models, our in-house translation that capture IKEA's unique tone of voice.

Neural Machine Translation (NMT) is a subfield of Artificial Intelligence and Machine Learning that focuses on the development of models that can automatically translate text from one natural language to another. NMT models are based on neural networks, which are a type of machine learning model that are designed to mimic the structure and function of the human brain.

<div align="center">
<img src="https://does.pasco.k12.fl.us/wp-content/uploads/does/2020/03/machine-translation-google.jpg" width="35%" height="35%">
</div>

NMT models typically use an encoder-decoder architecture, where the encoder processes the source text and compresses it into a fixed-length representation called a "context vector". The decoder then uses this context vector to generate the target text. The model is trained using a large dataset of parallel text, which consists of sentences or phrases in the source language and their corresponding translations in the target language.

![Neural network](https://miro.medium.com/max/720/1*BbF4o_uKCRKerXpZiJBlpg.webp)

The training process involves feeding the model with the source sentence and corresponding target sentence, the model then learns to predict the target sentence from the source sentence. During inference, the model takes in a source sentence and generates the most likely target sentence.

NMT models have been shown to be very effective at translating text, often producing translations that are more fluent and natural-sounding than those produced by traditional machine translation methods. These models are used in a variety of applications, including online translation services, automated localization of software and content, and assistive technology for non-native speakers. 

We have chosen an appropriate architecture that allows us to capture IKEA's unique tone of voice adapting a pre-trained model with IKEA's own data.

![Architecture](https://ars.els-cdn.com/content/image/1-s2.0-S2666651020300024-gr11.jpg)
