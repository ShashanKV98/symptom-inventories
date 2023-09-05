from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'adsfadfasfsfwdasd'
    app.config['UPLOAD_FOLDER'] = "./static/"

    from .views import views
    
    app.register_blueprint(views,url_prefix="/")

    # create_database(app)

    return app

