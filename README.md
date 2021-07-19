# Initializing a new Layer project
Clone this repo to start a new [Layer](http://layer.co/) project. 

Layer is a Declarative MLOps platform that empowers Data Science teams to implement end-to-end machine learning with minimal effort. Layer is a managed solution and can be used without the hassle involved in configuring and setting up servers. 

This repo contain a Layer project that you can quickly use to boostrap your machine learning projects. 

The empty Layer Project has the following files:
.
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
