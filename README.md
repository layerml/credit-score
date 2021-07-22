# What is Layer?
Layer is a [Declarative MLOps Platform](https://layer.co/) that empowers Data Science teams to implement end-to-end machine learning with minimal effort. Layer is a managed solution and can be used without the hassle involved in configuring and setting up servers. 


# Getting started
Clone this repo to start a new Layer project. 

Install Layer:
```
pip install layer-sdk
```

Clone this empty Layer Project:
```
layer clone https://github.com/layerml/empty
```

This repo contains a Layer project that you can quickly use to boostrap your machine learning projects. 

The empty Layer Project has the following files:
```
├── .layer
│   ├── project.yml                 # Main project configuration file
├── data
│   ├── features        
│   │   ├── dataset.yml             # Definition of your featureset
│   └── dataset         
│       └── dataset.yml             # Definition of the source data
└── models
    └── model           
        ├── model.yml               # Training directives of your model
        ├── model.py                # Definition of your model
        └── requirements.txt        # Environment config file, if required
```
