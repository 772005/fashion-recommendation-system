# 👗 Fashion Recommendation System
Fashion is an important part of our lives — what we wear often defines our style and personality. With thousands of clothing options available online, choosing the perfect outfit can be overwhelming.
To solve this, we created a Fashion Recommendation System that suggests similar clothing items based on an image you provide.
Instead of relying on your purchase history, our system looks at the visual features of a product (like color, shape, texture, etc.) using deep learning — helping you find visually similar outfits instantly.


## Introduction
Humans are inevitably drawn towards something that is visually more attractive. This tendency of 
humans has led to development of fashion industry over the course of time. With introduction of 
recommender systems in multiple domains, retail industries are coming forward with investments in 
latest technology to improve their business. Fashion has been in existence since centuries and will be 
prevalent in the coming days as well. Women are more correlated with fashion and style, and they 
have a larger product base to deal with making it difficult to take decisions. It has become an important 
aspect of life for modern families since a person is more often than not judged based on his attire. 
Moreover, apparel providers need their customers to explore their entire product line so they can 
choose what they like the most which is not possible by simply going into a cloth store.


## Related work
In the online internet era, the idea of Recommendation technology was initially introduced in the mid-90s. Proposed CRESA that combined visual features, textual attributes and visual attention of 
the user to build the clothes profile and generate recommendations. Utilized fashion magazines 
photographs to generate recommendations. Multiple features from the images were extracted to learn 
the contents like fabric, collar, sleeves, etc., to produce recommendations. In order to meet the 
diverse needs of different users, an intelligent Fashion recommender system is studied based on 
the principles of fashion and aesthetics. To generate garment recommendations, customer ratings and 
clothing were utilized in The history of clothes and accessories, weather conditions were 
considered in to generate recommendations.


##  Proposed methodology
In this project, we propose a model that uses Convolutional Neural Network and the Nearest 
neighbour backed recommender. As shown in the figure Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in 
inventory. The nearest neighbour’s algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

![Alt text](Screenshort/work-flow-model.png)


## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![Alt text](Screenshort/ResNet50%20Architecture.png)


## Getting the inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![Alt text](Screenshort/inventry-collection.png)


## Recommendation generation

To generate recommendations, our proposed approach uses Sklearn Nearest neighbours Oh Yeah. This allows us to find the nearest neighbours for the 
given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.


## Experiment and results

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. 
Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441
garment images. The networks are trained and validated on the dataset taken. The training results 
show a great accuracy of the model with low error, loss and good f-score.


### 🧺 Dataset Link

[Kaggle Dataset Big size 15 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)


### Simple App UI & output

![Alt text](Screenshort/webpage-interface.png)
![Alt text](Screenshort/webpage-interface-upload.png)
![Alt text](Screenshort/webpage-interface-recommend.png)


## Quick overview

- Input: an image uploaded by the user
- Processing: ResNet50 (feature extractor) -> L2-normalized embedding
- Search: NearestNeighbors (euclidean) over pre-computed embeddings
- Output: top 5 visually similar product images


## Prerequisites

- Python 3.8+ (use a virtual environment)
- Install dependencies from `requirements.txt`
- The precomputed files `image_features_embedding.pkl` and `img_files.pkl` must be present
  in the repository root (these are loaded by `main.py`).

If you don't have those .pkl files you can generate them with `train.py` (may require the dataset
and a GPU for speed).


## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
streamlit -> streamlit run main.py
flask -> python app.py
```

Open the provided URL in your browser, upload an image and the app will display the top 5
recommended images.


## Project layout

- `main.py`  - Streamlit app 
- `app.py`   - Flask app
- `train.py` - training and feature extraction script
- `test.py`  - additional utility scripts for testing
- `uploader/`- temporary storage for uploaded images (created at runtime)
- `Screenshort/`   - Contain images related to project  
- `fashion_small/` - dataset for the project
- `templates/` - contain html web page for flask
- `image_features_embedding.pkl`, `img_files.pkl` — required precomputed embeddings & file list


## License & contact

No license file is included. Add an appropriate license if you plan to reuse or redistribute.
For questions, open an issue in the repository or contact the maintainer.


## Conclusion

This project introduces a smart and visually driven fashion recommendation system.
By combining deep learning (ResNet50) and nearest neighbor search, the system can analyze an outfit image and recommend visually similar fashion products.

This makes online shopping easier, faster, and more personalized — helping users discover styles that truly match their preferences.
